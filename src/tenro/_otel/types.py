# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""OTel mapping types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MappedEvent:
    """Plain event representation.

    Attributes:
        name: Event name (e.g., gen_ai.evaluation.result).
        body: Event body as a plain dict.
    """

    name: str
    body: dict[str, Any]
