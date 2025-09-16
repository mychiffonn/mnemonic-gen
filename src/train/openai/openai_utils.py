"""Utility functions for OpenAI API interactions."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING

from structlog import getLogger

if TYPE_CHECKING:
    from typing import Literal, Optional

    from openai import OpenAI
    from structlog.stdlib import BoundLogger

    from src.utils.types import PathLike

from src.utils import check_file_path, read_config

logger: BoundLogger = getLogger(__name__)


def validate_openai_config(input_path: PathLike):
    """Validate the configuration file to be used for fine-tuning or generating completions using OpenAI.

    Args:
        input_path (PathLike): The path to the input configuration file. The configuration should be in JSON format.

    Raises:
        ValueError: If the input configuration is empty or not a dictionary.
        TypeError: If the input configuration is not a dictionary.
        IndexError: If the input configuration is missing required keys or has unrecognized keys.
    """
    input_path = check_file_path(input_path, extensions=["json"])

    config = read_config(input_path)

    # Check data
    if not config:
        logger.error(f"{input_path} is empty.")
        raise ValueError(f"{input_path} is empty.")
    elif not isinstance(config, dict):
        logger.error("Data cannot be read to a dictionary.")
        raise TypeError(
            f"Data cannot be read to a dictionary. Please review the data from {input_path}"
        )

    # Validate configuration
    required_keys = ["model"]
    optional_keys = [
        "temperature",
        "max_completion_tokens",
        "frequency_penalty",
        "presence_penalty",
        "top_p",
        "stream",
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Missing required keys: {missing_keys}")
        raise IndexError(f"Missing required keys: {missing_keys}")

    unrecognized_keys = [
        key for key in config if key not in required_keys + optional_keys
    ]
    if unrecognized_keys:
        logger.error(f"Unrecognized keys: {unrecognized_keys}")
        raise IndexError(f"Unrecognized keys: {unrecognized_keys}")

    # Add defaults for missing optional keys
    optional_defaults = {
        "temperature": 0.4,
        "max_completion_tokens": 2048,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 1.0,
        "stream": False,
    }

    for key, default in optional_defaults.items():
        if key not in config:
            config[key] = default

    return config


def validate_openai_file(input_path: PathLike) -> None:
    """Validate the data to be uploaded to OpenAI's API. Source code from OpenAI Cookbook: https://cookbook.openai.com/examples/chat_finetuning_data_prep.

    Args:
        input_path (PathLike): The path to the input data. The data should be in JSONL format.

    Raises:
        ValueError: If the input data is empty or not a list of dictionaries.
        TypeError: If the input data is not a list of dictionaries.
    """

    def validate_message(message: dict, format_errors: defaultdict) -> None:
        """Validate a single message."""
        if "role" not in message or (
            "content" not in message and "tool" not in message
        ):
            format_errors["message_missing_key"] += 1

        if any(
            k
            not in (
                "role",
                "content",
                "name",
                "weight",
                "refusal",
                "audio",
                "tool_calls",
                "tool_call_id",
            )
            for k in message
        ):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in (
            "developer",
            "system",
            "user",
            "assistant",
            "tool",
        ):
            format_errors["unrecognized_role"] += 1

        content = message.get("content")
        tool = message.get("tool")
        if (content is None and tool is None) or (
            content is not None and not isinstance(content, str)
        ):
            format_errors["missing_content"] += 1

    def validate_tools(tools: list, format_errors: defaultdict) -> None:
        """Validate tools in the dataset."""
        for tool in tools:
            if not isinstance(tool, dict):
                format_errors["tool_data_type"] += 1
                continue
            if any(k not in ("type", "function") for k in tool):
                format_errors["tool_unrecognized_key"] += 1

    input_path = check_file_path(input_path, extensions=["jsonl"])

    with input_path.open("r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    if not dataset:
        logger.error(f"{input_path} is empty.")
        raise ValueError(f"{input_path} is empty.")
    elif not isinstance(dataset, list) or not all(
        isinstance(ex, dict) for ex in dataset
    ):
        logger.error("Data cannot be read as a list of dictionaries.")
        raise TypeError(
            f"Data cannot be read as a list of dictionaries. Please review the data from {input_path}"
        )

    format_errors: defaultdict = defaultdict(int)

    for ex in dataset:
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
        else:
            for message in messages:
                validate_message(message, format_errors)

            if not any(msg.get("role") == "assistant" for msg in messages):
                format_errors["example_missing_assistant_message"] += 1

        tools = ex.get("tools", None)
        if tools:
            validate_tools(tools, format_errors)

    if format_errors:
        logger.exception(
            "Errors found in formatting data from input", input_path=input_path
        )
        for k, v in format_errors.items():
            logger.error(f"{k}: {v}")
    else:
        logger.info(
            f"Data from {input_path} is formatted ROUGHLY correctly. Always check the data manually with the OpenAI API reference here: https://platform.openai.com/docs/api-reference/fine-tuning/chat-input."
        )
        logger.info(f"Number of examples: {len(dataset)}")


def upload_file_to_openai(
    client: OpenAI,
    input_path: PathLike,
    purpose: Literal[
        "assistants", "batch", "fine-tune", "vision", "user_data", "evals"
    ] = "fine-tune",
) -> Optional[str]:
    """Upload the input file to OpenAI's Files API.

    Args:
        client (OpenAI): The OpenAI client object.
        input_path (PathLike): The path to the input file.
        purpose (str): The purpose of the file. Default is "fine-tune".

    Returns:
        Optional[str]: The id of the uploaded file.

    Raises:
        e: Exception if there was an error uploading the file.
    """
    try:
        # Validate the file path
        input_path = check_file_path(input_path, extensions=["jsonl"])

        with input_path.open("rb") as file_bin:
            logger.info("Uploading file to OpenAI", source=input_path)
            logger.debug("Type of file_bin object", type=type(file_bin))
            file_obj = client.files.create(file=file_bin, purpose=purpose)

        if file_obj is None:
            logger.error("Error uploading file: received None as file object.")
            raise Exception("Error uploading file: received None as file object.")
        if getattr(file_obj, "status", None) == "failed":
            logger.error(f"Upload failed: {file_obj.error}")
            raise Exception(f"Upload failed: {file_obj.error}")

        logger.info("File uploaded successfully", file_id=file_obj.id)
        logger.debug("File object details", file_obj=file_obj)
        return file_obj.id

    except Exception as e:
        logger.exception("Error during file upload")
        raise e
