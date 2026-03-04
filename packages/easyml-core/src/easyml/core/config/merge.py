"""Deep merge utility backed by OmegaConf."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge *overlay* into *base*, returning a new plain dict.

    - Nested dicts are merged recursively.
    - Lists are replaced wholesale (not appended).
    - Neither *base* nor *overlay* is mutated.

    Uses OmegaConf.merge() under the hood for reliable recursive merging.
    """
    base_cfg: DictConfig = OmegaConf.create(base)
    overlay_cfg: DictConfig = OmegaConf.create(overlay)
    merged: DictConfig = OmegaConf.merge(base_cfg, overlay_cfg)
    return OmegaConf.to_container(merged, resolve=True)
