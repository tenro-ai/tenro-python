# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Path formatting utilities."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path


def format_file_location(
    file_path: str,
    line: int,
    *,
    base: Path | None = None,
) -> str:
    """Format file:line location for error messages.

    Uses relative path if file is within base directory (matches mypy/ruff/pytest).
    Falls back to absolute path for files outside the base directory.

    Args:
        file_path: Absolute or relative path to file.
        line: Line number (1-indexed).
        base: Base directory for relative paths. Defaults to cwd.

    Returns:
        Formatted location string like "src/module.py:42".
    """
    if base is None:
        base = Path.cwd()
    with suppress(ValueError):
        file_path = str(Path(file_path).relative_to(base))
    return f"{file_path}:{line}"
