# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Gemini GenerateContent API Pydantic template.

Route: POST /v1/models/{model}:generateContent
Docs: https://ai.google.dev/api/rest/v1beta/GenerateContentResponse
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class GenerateContentResponse(BaseModel):
    """Gemini GenerateContentResponse schema.

    Forward-compatible template that validates core fields while
    allowing new fields added by Google.

    Schema from: https://ai.google.dev/api/rest/v1beta/GenerateContentResponse

    Attributes:
        candidates: Candidate response objects from the model.
    """

    candidates: list[dict[str, Any]]

    # Forward compatibility: Allow new fields from future API versions
    model_config = ConfigDict(extra="allow")
