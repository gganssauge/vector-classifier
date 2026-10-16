#!/usr/bin/env python3
"""CLI script for batch vector classification and offline audit pass."""

import sys

from vector_classifier.cli import app

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in [
        "audit",
        "version",
        "info",
        "run",
        "--help",
        "-h",
    ]:
        sys.argv.insert(1, "audit")
    elif len(sys.argv) == 1:
        sys.argv.append("audit")
    app()
