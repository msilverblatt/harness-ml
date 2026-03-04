from easyml.core.config.merge import deep_merge


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
    result = deep_merge(base, overlay)
    assert "c" not in base["a"]
