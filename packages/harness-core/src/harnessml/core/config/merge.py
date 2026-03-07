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


def resolve_feature_mutations(config: dict) -> dict:
    """Post-process merged config to apply append_features/remove_features.

    After ``deep_merge`` combines a base config with an overlay, any
    ``append_features`` or ``remove_features`` keys inside model definitions
    are inert — ``deep_merge`` does not know about them.  This function walks
    the ``models`` section and resolves those directives into the concrete
    ``features`` list, then removes the directive keys so the downstream
    schema never sees them.

    The input dict is mutated in place *and* returned for convenience.
    """
    models = config.get("models", {})
    if not isinstance(models, dict):
        return config
    for name, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            continue
        if "append_features" in model_cfg:
            existing = list(model_cfg.get("features", []))
            to_add = model_cfg.pop("append_features")
            model_cfg["features"] = existing + [f for f in to_add if f not in existing]
        if "remove_features" in model_cfg:
            existing = list(model_cfg.get("features", []))
            to_remove = model_cfg.pop("remove_features")
            model_cfg["features"] = [f for f in existing if f not in to_remove]
    return config
