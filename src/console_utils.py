"""
console_utils.py
================
Console output helpers.

These modules print progress with box-drawing characters, ✓ marks, and
units like W/m². On a legacy Windows console (cp1252) any such print
raises UnicodeEncodeError, which would otherwise crash the pipeline
partway through a run. Importing this module makes stdout/stderr
tolerant of those characters.
"""

import sys


def enable_utf8_output():
    """Reconfigure stdout/stderr to UTF-8, replacing unencodable characters.

    Safe to call repeatedly. No-ops on streams that don't support
    reconfiguration (e.g. when output is captured by a test harness).
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# Applied on import so that library modules which print unicode are safe
# no matter which entry point (CLI, notebook, test) pulls them in.
enable_utf8_output()
