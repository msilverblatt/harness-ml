"""Server context placeholder.

project_dir resolution is handled by individual handlers via
_common.resolve_project_dir() since some handlers (e.g., configure init)
need the raw string with allow_missing=True.
"""
