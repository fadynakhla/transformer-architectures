from typing import Type, TypeVar
import os
import re

import pydantic
import yaml


T = TypeVar("T", bound=pydantic.BaseModel)


def load_config(filepath: str, section: str, model_class: Type[T]) -> T:
    with open(filepath, "r") as file:
        config_dict = yaml.safe_load(file)

    try:
        section_data = config_dict[section]
    except KeyError:
        raise ValueError(f"Section '{section}' not found in yaml file.")

    section_str = yaml.dump(section_data)
    env_pattern = re.compile(r"\$\{([^}]+)\}")
    missing_env_vars = []

    def replace_env_var(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            missing_env_vars.append(var_name)
            return match.group(0)
        return value

    interpolated_section = env_pattern.sub(replace_env_var, section_str)

    if missing_env_vars:
        raise ValueError(f"Missing environment variables: {missing_env_vars}")

    section_data = yaml.safe_load(interpolated_section)

    try:
        return model_class(**section_data)
    except pydantic.ValidationError as e:
        raise ValueError(f"Validation failed for section '{section}': {e}")
