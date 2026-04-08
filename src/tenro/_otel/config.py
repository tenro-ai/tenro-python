# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""OTel export configuration.

Stores resource attributes for OTel span export. Config is set at init-time
and consumed lazily by the Construct.
"""

from __future__ import annotations

resource_attributes: dict[str, str] | None = None
"""Extra OTel resource attributes set by ``tenro.init()``."""
