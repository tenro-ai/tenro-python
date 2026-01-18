# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for examples.

The construct fixture is auto-provided by tenro.pytest_plugin.
No additional configuration needed - the plugin registers fixtures automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Avoid shadowing third-party framework packages (e.g. `examples/pydantic_ai`).
_EXAMPLES_DIR = str(Path(__file__).resolve().parent)
if _EXAMPLES_DIR in sys.path:
    sys.path.remove(_EXAMPLES_DIR)
