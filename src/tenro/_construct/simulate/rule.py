# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Simulation rule model for Construct simulation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

SimulationStrategy = Literal["dispatch", "code_swap", "http"]
"""Strategy used for simulation.

dispatch: Call-time dispatch (linked callables via @link_*)
code_swap: __code__ patching (registered unlinked callables)
http: HTTP boundary interception (LLM providers)
"""


class SimulationRule(BaseModel):
    """Validated rule for simulation data.

    Enforces mutual exclusivity between static values and side effects.

    Stores configuration for one simulation target, including the response
    strategy (static value, callback, or sequence) and dispatcher metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    returns_value: object | None = None
    side_effect: object | None = None
    result_sequence: list[object] | None = None
    generator_items: list[object] | None = None
    is_linked: bool = False
    is_generator: bool = False
    strategy: SimulationStrategy = "dispatch"
    llm_provider: str | None = None  # For LLM simulations: provider to mark triggered
    llm_model: str | None = None  # For LLM simulations: model to display in trace

    @model_validator(mode="before")
    @classmethod
    def check_exclusive_fields(cls, values: dict[str, object]) -> dict[str, object]:
        """Ensure return values and side effects are mutually exclusive.

        Args:
            values: Raw field values for validation.

        Returns:
            The validated field mapping.

        Raises:
            ValueError: If both returns_value and side_effect are provided.

        Examples:
            >>> SimulationRule(returns_value="ok")
            SimulationRule(returns_value='ok', side_effect=`None`)
        """
        if values.get("returns_value") is not None and values.get("side_effect") is not None:
            raise ValueError("Cannot set both 'returns_value' and 'side_effect'")
        return values
