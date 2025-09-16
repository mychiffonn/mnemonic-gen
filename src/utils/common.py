"""Module for common utility functions."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from structlog import getLogger

from src.constants import PROMPT_PATH, BasePath
from src.utils.error_handlers import check_dir_path, check_file_path

if TYPE_CHECKING:
    from re import Pattern
    from typing import Any, Optional, Union

    from src.utils import PathLike

logger = getLogger(__name__)


def search_files(
    search_path: PathLike, regex_pattern: str | Pattern[str]
) -> list[Path]:
    """Search for files in a directory that match a regex pattern.

    Args:
        search_path (PathLike): The path to the directory to search.
        regex_pattern (str): The regex pattern to match filenames.

    Returns:
        list[PathLike]: A list of file paths that match the regex pattern.
    """
    search_path = check_dir_path(search_path, new_ok=False)

    matching_files = []
    # Loop recursively through all files in the directory
    for file in search_path.rglob("*"):
        if file.is_file() and re.search(regex_pattern, file.name):
            matching_files.append(file)

    if not matching_files:
        logger.debug(
            "No files found matching the regex pattern",
            search_path=search_path,
            regex_pattern=regex_pattern,
        )
        raise ValueError(
            f"No files found in '{search_path}' matching the regex pattern: '{regex_pattern}'"
        )

    return matching_files


def get_first_prompt_file(regex_pattern: str | Pattern[str]) -> Path:
    """Search for prompt files in the prompts directory that match a regex pattern.

    Args:
        regex_pattern (str): The regex pattern to match filenames.

    Returns:
        Path: The first file path that matches the regex pattern.
    """
    matching_files = search_files(BasePath.PROMPTS, regex_pattern)

    if not matching_files:
        logger.debug(
            "No prompt files found matching the regex pattern",
            regex_pattern=regex_pattern,
        )
        raise ValueError(
            f"No prompt files found in {BasePath.PROMPTS} matching the regex pattern: {regex_pattern}"
        )

    return matching_files[0]


def get_first_config_file(regex_pattern: str | Pattern[str]) -> Path:
    """Search for configuration files in the config directory that match a regex pattern.

    Args:
        regex_pattern (str): The regex pattern to match filenames.

    Returns:
        Path: The first file path that matches the regex pattern.
    """
    matching_files = search_files(BasePath.CONFIG, regex_pattern)

    if not matching_files:
        logger.debug(
            "No config files found matching the regex pattern",
            regex_pattern=regex_pattern,
        )
        raise ValueError(
            f"No config files found in {BasePath.CONFIG} matching the regex pattern: {regex_pattern}"
        )

    return matching_files[0]


def read_prompt(
    prompt_path: Optional[PathLike] = None,
    regex_pattern: Optional[Union[str, Pattern[str]]] = None,
    vars: Optional[dict[str, Any]] = None,
    vars_json_path: Optional[PathLike] = None,
) -> str:
    """Read the system prompt from a .txt file.

    Args:
        prompt_path (PathLike, optional): The path to the prompt file.
        regex_pattern (str, optional): The regex pattern to match the prompt file. Default is r'*.txt$'.
        vars (dict, optional): A dictionary of variables to replace in the prompt.
        vars_json_path (PathLike, optional): The path to a JSON file containing variables.

    Returns:
        str: The prompt.
    """
    if prompt_path is None and regex_pattern is None:
        raise ValueError(
            "Either prompt_path or regex_pattern must be provided to read a prompt."
        )
    elif prompt_path:
        prompt_path_obj: Path = check_file_path(prompt_path, extensions=["txt"])
    elif regex_pattern:
        prompt_path_obj: Path = get_first_prompt_file(regex_pattern)

    logger.debug("Reading prompt from file", source=prompt_path)

    with prompt_path_obj.open("r") as file:
        prompt = file.read().strip()

    if vars_json_path:
        vars_from_json = read_config(vars_json_path)

        # If vars is also provided, extend it with the values from the JSON file
        # else use the values from the JSON file as the vars
        if vars:
            vars.update(vars_from_json)
        else:
            vars = vars_from_json

    elif vars_json_path is None and (
        "system" in prompt_path_obj.name and "_" not in prompt_path_obj.name
    ):
        vars = read_config(PROMPT_PATH.PLACEHOLDER_DICT)

    if vars:
        logger.debug(
            "Substituting these variables in config file", source=prompt_path, vars=vars
        )
        return prompt.format(**vars)

    return prompt


def sample_prompts(prompt_path: PathLike, num_samples: int = 1) -> str | list[str]:
    """Sample random instruction(s) from a .txt file.

    Args:
        prompt_path (PathLike): The path to the file (.txt) with prompts
        num_samples (int): The number of random prompts to return. Default is 1.

    Returns:
        str: A random instruction from the file.
    """
    prompt_path = check_file_path(prompt_path, extensions=[".txt"])

    with prompt_path.open("r") as file:
        prompts = file.readlines()

    # Strip whitespace and remove empty lines
    prompts = [line.strip() for line in prompts if line.strip()]

    if not prompts:
        raise ValueError(f"No valid prompts found in {prompt_path}")

    if num_samples == 1:
        chosen_prompt = random.choice(prompts)
        logger.debug(
            "Sampled a single prompt", source=prompt_path, prompt=chosen_prompt
        )
        return chosen_prompt
    else:
        if num_samples > len(prompts):
            raise ValueError(
                f"Requested {num_samples} samples, but only {len(prompts)} available."
            )

        chosen_prompts = random.sample(prompts, num_samples)
        logger.debug(
            "Sampling multiple prompts",
            source=prompt_path,
            prompts=chosen_prompts,
        )
        return chosen_prompts


def read_config(
    conf_path: Optional[PathLike] = None,
    regex_pattern: Optional[Union[str | Pattern[str]]] = None,
) -> dict:
    """Read a configuration file.

    Args:
        conf_path (PathLike): The path to the configuration file. Must be a JSON file.
        regex_pattern (str): The regex pattern to search for config files in the directory.

    Returns:
        dict: The configuration.
    """
    if conf_path is None and regex_pattern is None:
        raise ValueError(
            "Either conf_path or regex_pattern must be provided to read a configuration file."
        )
    elif conf_path:
        # Convert to Path object, ensure the file path exists and has the correct extension
        conf_path_obj = check_file_path(
            conf_path, extensions=[".json", ".yaml", ".yml"]
        )
    elif regex_pattern:
        conf_path_obj = get_first_config_file(regex_pattern)

    if conf_path_obj.suffix == ".json":
        with conf_path_obj.open("r") as file:
            return json.load(file)
    elif conf_path_obj.suffix in [".yaml", ".yml"]:
        with conf_path_obj.open("r") as file:
            return yaml.safe_load(file) or {}
    else:
        raise ValueError(
            f"Unsupported file format: {conf_path_obj.suffix}. Only .json and .yaml/.yml are supported."
        )


def update_config(config_filepath: PathLike, key: str, new_value: Any):
    """Update the config file with the new_value for the key.

    Args:
        config_filepath (PathLike): The path to the config file. The file should be in JSON format.
        key (str): The key to update.
        new_value (Any): The new value to set for the key.
    """
    config_path = check_file_path(config_filepath, extensions=["json"])
    try:
        config_data: dict = read_config(config_path)
        config_data[key] = new_value
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f)
    except Exception as e:
        raise e
