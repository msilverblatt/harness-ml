"""Main entry point for resolving multi-file, variant-aware configs."""

from __future__ import annotations

from pathlib import Path

from harnessml.core.config.loader import load_config_file
from harnessml.core.config.merge import deep_merge


def resolve_config(
    config_dir: str | Path,
    file_map: dict[str, str | list[str]],
    variant: str | None = None,
    overlay: dict | None = None,
) -> dict:
    """Load, merge, and overlay a split config directory into a single dict.

    Parameters
    ----------
    config_dir:
        Root directory containing all config files.
    file_map:
        Mapping of logical section name to filename(s) relative to
        *config_dir*.  A string value loads one file; a list of strings
        loads each file in order and deep-merges them together.
    variant:
        Optional variant suffix (e.g. ``"w"`` for women's).  Each file
        in *file_map* is resolved through :func:`load_config_file`'s
        variant logic.
    overlay:
        Optional dict deep-merged on top of the final result.

    Returns
    -------
    dict
        Fully resolved configuration.
    """
    config_dir = Path(config_dir)
    result: dict = {}

    for _section, filenames in file_map.items():
        if isinstance(filenames, str):
            filenames = [filenames]

        section_merged: dict = {}
        for fname in filenames:
            loaded = load_config_file(config_dir, fname, variant=variant)
            section_merged = deep_merge(section_merged, loaded)

        result = deep_merge(result, section_merged)

    if overlay is not None:
        result = deep_merge(result, overlay)

    return result
