from harnessml.core.config.merge import deep_merge, resolve_feature_mutations


def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    overlay = {"b": 3, "c": 4}
    result = deep_merge(base, overlay)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"models": {"xgb": {"params": {"depth": 3, "lr": 0.05}}}}
    overlay = {"models": {"xgb": {"params": {"depth": 5}}}}
    result = deep_merge(base, overlay)
    assert result["models"]["xgb"]["params"]["depth"] == 5
    assert result["models"]["xgb"]["params"]["lr"] == 0.05


def test_deep_merge_list_replaces():
    base = {"features": ["a", "b", "c"]}
    overlay = {"features": ["x", "y"]}
    result = deep_merge(base, overlay)
    assert result["features"] == ["x", "y"]


def test_deep_merge_new_key():
    base = {"a": 1}
    overlay = {"b": 2}
    result = deep_merge(base, overlay)
    assert result == {"a": 1, "b": 2}


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    overlay = {"a": {"c": 2}}
    deep_merge(base, overlay)
    assert "c" not in base["a"]


# --- resolve_feature_mutations tests ---


def test_resolve_feature_mutations_append():
    config = {
        "models": {
            "xgb_main": {
                "features": ["a", "b", "c"],
                "append_features": ["d", "e"],
            }
        }
    }
    result = resolve_feature_mutations(config)
    assert result["models"]["xgb_main"]["features"] == ["a", "b", "c", "d", "e"]
    assert "append_features" not in result["models"]["xgb_main"]


def test_resolve_feature_mutations_remove():
    config = {
        "models": {
            "xgb_main": {
                "features": ["a", "b", "c", "d"],
                "remove_features": ["b", "d"],
            }
        }
    }
    result = resolve_feature_mutations(config)
    assert result["models"]["xgb_main"]["features"] == ["a", "c"]
    assert "remove_features" not in result["models"]["xgb_main"]


def test_resolve_feature_mutations_append_and_remove():
    config = {
        "models": {
            "xgb_main": {
                "features": ["a", "b", "c"],
                "append_features": ["d", "e"],
                "remove_features": ["b", "d"],
            }
        }
    }
    result = resolve_feature_mutations(config)
    # append first: [a, b, c, d, e], then remove b and d: [a, c, e]
    assert result["models"]["xgb_main"]["features"] == ["a", "c", "e"]
    assert "append_features" not in result["models"]["xgb_main"]
    assert "remove_features" not in result["models"]["xgb_main"]


def test_resolve_feature_mutations_no_duplicates_on_append():
    config = {
        "models": {
            "xgb_main": {
                "features": ["a", "b"],
                "append_features": ["b", "c"],
            }
        }
    }
    result = resolve_feature_mutations(config)
    assert result["models"]["xgb_main"]["features"] == ["a", "b", "c"]


def test_resolve_feature_mutations_no_models():
    config = {"ensemble": {"method": "weighted"}}
    result = resolve_feature_mutations(config)
    assert result == {"ensemble": {"method": "weighted"}}


def test_resolve_feature_mutations_no_directives():
    config = {
        "models": {
            "xgb_main": {"features": ["a", "b"], "params": {"depth": 3}}
        }
    }
    result = resolve_feature_mutations(config)
    assert result["models"]["xgb_main"]["features"] == ["a", "b"]


def test_resolve_feature_mutations_with_deep_merge():
    """End-to-end: deep_merge overlay then resolve mutations."""
    base = {
        "models": {
            "xgb_main": {
                "features": ["a", "b", "c"],
                "params": {"depth": 3},
            }
        }
    }
    overlay = {
        "models": {
            "xgb_main": {
                "append_features": ["d"],
                "remove_features": ["a"],
                "params": {"depth": 5},
            }
        }
    }
    merged = deep_merge(base, overlay)
    result = resolve_feature_mutations(merged)
    assert result["models"]["xgb_main"]["features"] == ["b", "c", "d"]
    assert result["models"]["xgb_main"]["params"]["depth"] == 5
    assert "append_features" not in result["models"]["xgb_main"]
    assert "remove_features" not in result["models"]["xgb_main"]
