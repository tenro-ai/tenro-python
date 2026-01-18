# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Environment utilities for Tenro SDK."""

from __future__ import annotations

import os

from tenro.errors import TenroValidationError

_TRUTHY_VALUES = ("true", "1", "yes", "on")
_FALSY_VALUES = ("false", "0", "no", "off")


def get_env_var(name: str, required: bool = False) -> str | None:
    """Get environment variable with optional validation.

    Args:
        name: Environment variable name.
        required: If True, raise TenroValidationError if not set.

    Returns:
        Environment variable value or None.

    Raises:
        ValidationError: If required=True and variable not set.
    """
    value = os.getenv(name)
    if required and not value:
        raise TenroValidationError(f"Required environment variable '{name}' is not set")
    return value


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get a boolean environment variable with a fallback.

    Args:
        name: Environment variable name.
        default: Value when unset or unrecognized.

    Returns:
        Parsed boolean value.
    """
    value = os.getenv(name)
    if not value:
        return default

    normalized = value.strip().lower()
    if normalized in _TRUTHY_VALUES:
        return True
    if normalized in _FALSY_VALUES:
        return False
    return default
