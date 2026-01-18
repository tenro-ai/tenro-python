# Copyright 2025 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Linking utilities for agents, LLMs, and tools.

Provides decorators for registering functions, classes, and framework objects
with the Tenro system for testing and observability.
"""

from __future__ import annotations

from tenro.linking.context import (
    get_active_construct as _get_active_construct,
)
from tenro.linking.context import (
    set_active_construct as _set_active_construct,
)
from tenro.linking.decorators import link_agent, link_llm, link_tool
from tenro.linking.metadata import (
    classify_target as _classify_target,
)
from tenro.linking.metadata import (
    find_entry_methods as _find_entry_methods,
)
from tenro.linking.metadata import (
    get_display_name as _get_display_name,
)
from tenro.linking.metadata import (
    get_target_paths as _get_target_paths,
)
from tenro.linking.metadata import (
    is_directly_linked as _is_directly_linked,
)

__all__ = [
    "_classify_target",
    "_find_entry_methods",
    "_get_active_construct",
    "_get_display_name",
    "_get_target_paths",
    "_is_directly_linked",
    "_set_active_construct",
    "link_agent",
    "link_llm",
    "link_tool",
]
