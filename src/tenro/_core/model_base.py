# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Base Pydantic models for Tenro SDK."""

from __future__ import annotations

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
        validate_assignment=True,
    )
