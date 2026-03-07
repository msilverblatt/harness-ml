def test_package_exports():
    from harnessml.core.config import deep_merge, load_config_file, resolve_config

    assert callable(deep_merge)
    assert callable(load_config_file)
    assert callable(resolve_config)
