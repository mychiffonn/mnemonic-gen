"""Script for training Gemma-3 models with DPO and LoRA adapters."""

import os
from pathlib import Path

import torch
import wandb
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel
from structlog import getLogger
from trl import DPOConfig, DPOTrainer

from unsloth import FastModel  # isort: skip

from src.utils.common import read_config
from src.utils.hf_utils import get_hf_token, login_hf_hub

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = getLogger(__name__)

# Load config
config = read_config("config/train/dpo.yaml")

# LoRA and model config
hf_username = "chiffonng"
hf_base_model = "unsloth/gemma-3-7b"
max_seq_length = 1024
lora_rank = 16

# Config models


class DataConfigModel(BaseModel):
    """Data configuration."""

    train_dataset: str
    test_dataset: str


class PushConfigModel(BaseModel):
    """Push configuration."""

    repo_id: str
    gguf: dict
    merged: dict


class ConfigModel(BaseModel):
    """Training configuration."""

    user: str
    base_model: str
    train: DPOConfig
    data: DataConfigModel
    push: PushConfigModel


# Load config
def load_config(config_path: str | Path) -> ConfigModel:
    """Load config from YAML file."""
    with Path(config_path).open("r") as f:
        config_dict = yaml.safe_load(f)
    return ConfigModel(**config_dict)


load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(
    project=f"{hf_base_model}-links-dpo",
    job_type="training",
    anonymous="allow",
)
logger.info("WandB initialized")
login_hf_hub()

# Load datasets
required_columns = ["prompt", "chosen", "rejected"]
logger.info("Loading datasets")
train_dataset = load_dataset(config.data.train_dataset, split="train")
val_dataset = load_dataset(config.data.train_dataset, split="val")
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in required_columns]
)
val_dataset = val_dataset.remove_columns(
    [col for col in val_dataset.column_names if col not in required_columns]
)

# Setup model with LoRA
logger.info(f"Loading base model: {hf_base_model}")
model, tokenizer = FastModel.from_pretrained(
    model_name=hf_base_model,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=lora_rank,
    lora_alpha=2 * lora_rank,
    lora_dropout=0,
    bias="none",
    random_state=42,
    use_rslora=True,
)

# DPO training config
# Create DPOConfig from our config model
training_args = DPOConfig(
    **config.train.model_dump(),
    bf16=True,
    # Save strategy
    run_name=run.id,
)

# Create and run the DPO trainer
logger.info("Setting up DPO trainer")
trainer = DPOTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
)

logger.info("Starting training")
trainer.train()
logger.info("Training complete")

# Push models to HuggingFace Hub for vllm
logger.info("Pushing models to HuggingFace Hub")
model.push_to_hub_merged(
    f"{hf_username}/{hf_base_model}-links-dpo",
    tokenizer,
    save_method=config.push.merged["save_methods"][0],
    hf_token=get_hf_token(),
)

# Push GGUF model
logger.info("Pushing GGUF model")
model.push_to_hub_gguf(
    f"{hf_username}/{hf_base_model}-links-gguf",
    tokenizer,
    quantization_method=config.push["quantization_methods"],
    token=get_hf_token(),
)

logger.info("All done!")
wandb.finish()
