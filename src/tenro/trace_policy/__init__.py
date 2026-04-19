# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Trace policy — controls what span content is captured and how.

Three modes: ``off`` (default), ``full``, and ``custom``. Six content
groups (``inputs``, ``outputs``, ``tool_content``, ``media``, ``embeddings``,
``metadata``) can be independently toggled. A ``custom`` policy delegates
to a user-supplied callback; helpers in ``tenro.trace_policy.transforms``
cover the common cases (hashing, pseudonymization, truncation).

Examples:
    Default (off — content drops, structural data emits):

    >>> import tenro
    >>> tenro.configure_trace_policy()
    TracePolicy(capture=<CaptureMode.OFF: 'off'>, ...)

    Capture everything (dev or test):

    >>> tenro.configure_trace_policy(capture="full")
    TracePolicy(capture=<CaptureMode.FULL: 'full'>, ...)

    Capture outputs but not inputs:

    >>> tenro.configure_trace_policy(
    ...     capture="full",
    ...     groups={"inputs": False},
    ... )
    TracePolicy(...)

    Hash inputs with a project-scoped key:

    >>> from tenro.trace_policy import CaptureMode
    >>> from tenro.trace_policy.transforms import keyed_blake2b
    >>> hash_fn = keyed_blake2b(key=b"project-secret")
    >>> def mask(value, ctx):
    ...     return hash_fn(value, ctx) if ctx.category.value == "inputs" else value
    >>> tenro.configure_trace_policy(
    ...     capture=CaptureMode.CUSTOM,
    ...     transform=mask,
    ... )
    TracePolicy(...)
"""

from __future__ import annotations

from tenro.trace_policy.applier import CaptureResult, apply
from tenro.trace_policy.categories import Category
from tenro.trace_policy.config import (
    configure_trace_policy,
    get_policy,
    override,
    reset,
)
from tenro.trace_policy.env import (
    ENV_GROUPS,
    ENV_POLICY,
    TenroTracePolicyConfigError,
    TenroTracePolicyError,
)
from tenro.trace_policy.modes import (
    OMIT,
    CaptureMode,
    Transform,
    TransformContext,
)
from tenro.trace_policy.policy import POLICY_VERSION, TracePolicy

__all__ = [
    "ENV_GROUPS",
    "ENV_POLICY",
    "OMIT",
    "POLICY_VERSION",
    "CaptureMode",
    "CaptureResult",
    "Category",
    "TenroTracePolicyConfigError",
    "TenroTracePolicyError",
    "TracePolicy",
    "Transform",
    "TransformContext",
    "apply",
    "configure_trace_policy",
    "get_policy",
    "override",
    "reset",
]
