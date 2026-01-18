# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle management for Construct spans."""

from tenro._construct.lifecycle.linker import SpanLinker
from tenro._construct.lifecycle.span_accessor import SpanAccessor

__all__ = ["SpanAccessor", "SpanLinker"]
