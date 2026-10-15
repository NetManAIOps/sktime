#!/usr/bin/env python3
"""Deprecated alias: the CLI is now `libraryts` (playground/libraryts.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libraryts import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
