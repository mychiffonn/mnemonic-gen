# mypy: disable-error-code=union-attr
"""Module for processing mnemonic data using OpenAI's API.

Finetuning: https://platform.openai.com/docs/guides/fine-tuning
Files API: https://platform.openai.com/docs/api-reference/files
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError
from structlog import getLogger

from src import constants as const
from src.data_prep.data_io import read_csv_file, write_jsonl_file
from src.train.openai import (
    upload_file_to_openai,
    validate_openai_file,
)
from src.utils.common import read_config, read_prompt, update_config
from src.utils.error_handlers import check_file_path, check_file_paths

if TYPE_CHECKING:
    from typing import Any, Literal, Optional

    from openai import OpenAI
    from structlog.stdlib import BoundLogger

    from src.utils.types import ModelType, PathLike

# Set up logging
logger: BoundLogger = getLogger(__name__)


def prepare_finetune_data(
    input_path: PathLike,
    system_prompt_path: PathLike,
    user_prompt_path: PathLike,
) -> list[dict[str, Any]]:
    """Prepare the data (JSONL) for fine-tuning with OpenAI's API and split into training and validation files.

    The output file is in JSONL format, where each line is a JSON object with the following structure:
       {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}, {"role": "assistant", "content": assistant_response}]}

    Args:
        input_path (Path): The path to the input data in CSV format. To be formatted as assistant response.
        system_prompt_path (Path): The path to the system prompt file.
        user_prompt_path (Path): The path to the user prompt file.

    Returns:
        list[dict[str, Any]]: The formatted data, ready to be saved as JSONL.

    Raises:
         ValueError: If the split ratio is invalid.
    """
    # Validate paths
    input_path = check_file_path(input_path, extensions=const.Extension.CSV)
    system_prompt_path, user_prompt_path = check_file_paths(
        system_prompt_path, user_prompt_path, extensions=const.Extension.TXT
    )

    # Read prompts
    try:
        system_prompt = read_prompt(
            system_prompt_path, vars_json_path=const.PROMPT_PATH.PLACEHOLDER_DICT
        )
        logger.debug(
            "Read system prompt",
            prompt=f"{system_prompt[:100]} + ..."
            if len(system_prompt) > 100
            else system_prompt,
        )
    except Exception as e:
        logger.exception("Error reading system prompt")
        raise ValueError(f"Error reading system prompt: {e}") from e
    try:
        user_prompt_template = read_prompt(user_prompt_path)
        logger.debug("Read user prompt into template", prompt=user_prompt_template)
    except Exception as e:
        logger.exception("Error reading user prompt template")
        raise ValueError(f"Error reading user prompt template: {e}") from e

    # Read CSV file, convert to JSONL, and validate schema
    try:
        rows = read_csv_file(input_path, to_list=True)
        logger.debug(f"Read {len(rows)} rows from {input_path} using pandas")

        if not rows:
            raise ValueError(f"No data found in {input_path}")

        # Convert rows to OpenAI fine-tuning format
        formatted_rows = []
        for row in rows:
            # Format the user prompt with the term and mnemonic
            user_content = user_prompt_template.format(
                term=row.get("term", ""), mnemonic=row.get("mnemonic", "")
            )

            # Create the assistant response by excluding term and mnemonic from row
            assistant_content = {
                k: v for k, v in row.items() if k not in ["term", "mnemonic"]
            }
            # Create the message format for OpenAI fine-tuning
            formatted_row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": str(assistant_content)},
                ]
            }
            formatted_rows.append(formatted_row)

        return formatted_rows

    except ValidationError as e:
        logger.exception("Validation error while processing content file")
        raise e

    except Exception as e:
        logger.exception("Error processing content file")
        raise ValueError(f"Error processing content file: {e}") from e


def split_export_finetune_data(
    data: list[dict[str, Any]],
    output_train_jsonl: PathLike,
    output_val_jsonl: Optional[PathLike] = None,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[PathLike, Optional[PathLike]]:
    """Split data into training and validation sets and save to JSONL files.

    Args:
        data: List of dictionaries formatted for OpenAI finetuning.
        output_train_jsonl: Path to save the training data.
        output_val_jsonl: Path to save the validation data. If None, the validation data will not be saved.
        split_ratio: Ratio of training data to total data. Must be between 0 and 1. Default is 0.8.
        seed: Random seed for shuffling.

    Returns:
        tuple: Paths to the training and validation JSONL files.
    """
    # Validate the output_train_jsonl path
    output_train_jsonl = check_file_path(
        output_train_jsonl,
        new_ok=True,
        to_create=True,
        extensions=const.Extension.JSONL,
    )

    # Validate the output_val_jsonl path
    if output_val_jsonl is None:
        logger.info(
            "There is no validation data path provided. Validation data will not be saved."
        )
    else:
        output_val_jsonl = check_file_path(
            output_val_jsonl,
            extensions=const.Extension.JSONL,
            new_ok=True,
            to_create=True,
        )

    # Validate the split ratio
    if not isinstance(split_ratio, float):
        split_ratio = float(split_ratio)
    if split_ratio < 0 or split_ratio > 1:
        logger.exception("Invalid split ratio. Must be between 0 and 1.")
        raise ValueError(f"Invalid split ratio: {split_ratio}. Must be between 0 and 1")

    # TODO: Stratified split
    if output_val_jsonl:
        # Calculate split point
        split_idx = int(len(data) * split_ratio)
        train_rows = data[:split_idx]
        val_rows = data[split_idx:]

        # Log the number of training and validation rows
        num_train = len(train_rows)
        num_val = len(val_rows)
        logger.info(
            "Split data into training and validation sets",
            num_train=num_train,
            num_val=num_val,
        )

        # Write to JSONL files and validate
        write_jsonl_file(data=train_rows, file_path=output_train_jsonl)
        write_jsonl_file(data=val_rows, file_path=output_val_jsonl)
        validate_openai_file(output_train_jsonl)
        validate_openai_file(output_val_jsonl)

        logger.info(
            "Validation data saved to JSONL files",
            val_file=output_val_jsonl,
            num_val=num_val,
        )
    else:
        train_rows = data
        logger.info("No validation data path provided. Using all data for training.")
        write_jsonl_file(data=train_rows, file_path=output_train_jsonl)
        validate_openai_file(output_train_jsonl)
        logger.info(
            "Training data saved to JSONL file",
            train_file=output_train_jsonl,
            num_train=len(train_rows),
        )

    return output_train_jsonl, output_val_jsonl


def upload_finetune_data(
    client: OpenAI,
    input_path: PathLike,
    config_file_path: PathLike,
    file_type: Optional[Literal["training_file", "validation_file"]] = "training_file",
    to_reupload: bool = False,
) -> str | None:
    """Upload the fine-tuning data to OpenAI's Files API, reusing a cached file id if available.

    Args:
        client (OpenAI): The OpenAI client object.
        input_path (PathLike): Path to the JSONL input data.
        config_file_path (PathLike): Path to the JSON config file.
        file_type (str, optional): The type of file to upload (e.g., "train", "val", "training", "valid", "validation"). Defaults to "training_file".
        to_reupload (bool): If True, delete the cached file id from config and re-upload the file.

    Returns:
        str: The file id of the uploaded file, or None if there was an error.
    """
    # Ensure input_path is valid (expects a .jsonl file)
    input_path = check_file_path(input_path, extensions=["jsonl"])

    if (
        file_type is None
        or isinstance(file_type, str)
        or file_type != "training_file"
        or file_type != "validation_file"
    ):
        logger.warning(
            "File type is None or not specified. Attempting to infer from input path."
        )

    if "train" in str(input_path).lower():
        file_type = "training_file"
    elif "val" in str(input_path).lower():
        file_type = "validation_file"
    else:
        logger.warning(
            "File type could not be inferred from input path. Defaulting to 'training_file'."
        )
        file_type = "training_file"

    # Log ALL the argument values
    logger.debug(
        "Show all arguments",
        input_path=input_path,
        config_file_path=config_file_path,
        file_type=file_type,
        to_reupload=to_reupload,
    )
    logger.debug(
        "Current OpenAI finetune FILES", files=client.files.list(purpose="fine-tune")
    )

    # Attempt to use the cached file id if allowed.
    if not to_reupload:
        cached_file_id = _get_cached_file_id(
            client, config_file_path, file_type, to_reupload=to_reupload
        )
        if cached_file_id:
            return cached_file_id
    else:
        # If overwriting, delete the cached file id.
        _get_cached_file_id(
            client, config_file_path, file_type, to_reupload=to_reupload
        )

    # Upload the file since no valid cache was found.
    new_file_id = upload_file_to_openai(client, input_path=input_path)
    if new_file_id:
        update_config(config_file_path, key=file_type, new_value=new_file_id)
        logger.info(
            "Uploaded file for finetuning to OpenAI",
            file_id=new_file_id,
            source=input_path,
        )
        return new_file_id

    logger.debug(
        "Current OpenAI finetune FILES", files=client.files.list(purpose="fine-tune")
    )

    return None


def _get_cached_file_id(
    client: OpenAI, config_file_path: PathLike, file_type: str, to_reupload: bool
) -> str | None:
    """Retrieve the cached file ID for the specified file type from the config file.

    For the given file type (e.g. "training_file" or "validation_file"), verify that the cached
    file exists via OpenAI's Files API. If to_overwrite is True, delete the file from the API.

    Args:
        client (OpenAI): The OpenAI client object.
        config_file_path (Path): Path to the JSON config file.
        file_type (str): The key in the config (e.g., "training_file", "validation_file").
        to_reupload (bool): If True, delete the cached file from the API.

    Returns:
        str | None: The valid cached file id, or None if not found or if deleted.
    """

    def _check_and_handle_file(file_id: str) -> str | None:
        """Helper function to check if the file exists in the API and handle it accordingly."""
        try:
            file_info = client.files.retrieve(file_id)
            if file_info:
                if to_reupload:
                    client.files.delete(file_id)
                    logger.debug("Deleted cached file id", file_id=file_id)
                    return None
                else:
                    logger.debug("Using cached file id:", file_id=file_id)
                    return file_id
            return None
        except Exception:
            logger.exception("Error retrieving file", file_id=file_id)
            return None

    # Read the config file and get the file id
    try:
        config_file_path = check_file_path(
            config_file_path, extensions=const.Extension.JSON
        )
        config_data = read_config(config_file_path)
        candidate = config_data.get(file_type, "").strip()
        if candidate:
            return _check_and_handle_file(candidate)
        else:
            return None
    except Exception as e:
        logger.exception("Error reading config file:", config_file=config_file_path)
        raise e
