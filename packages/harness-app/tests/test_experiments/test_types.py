from harness.app.experiments.types import ExperimentType, EXPERIMENT_CONFIG_MAP


def test_all_experiment_types_exist():
    expected = {
        "BASELINE", "FEATURE", "MODEL", "HYPERPARAMETER",
        "ENSEMBLE", "CALIBRATION", "CV_STRATEGY", "FEATURE_SELECTION", "DATA_REFRESH",
    }
    actual = {member.name for member in ExperimentType}
    assert actual == expected


def test_config_map_has_entry_for_each_type():
    for exp_type in ExperimentType:
        assert exp_type in EXPERIMENT_CONFIG_MAP, f"Missing config map entry for {exp_type}"


def test_enum_values_are_lowercase_strings():
    for member in ExperimentType:
        assert isinstance(member.value, str), f"{member.name} value is not a string"
        assert member.value == member.value.lower(), f"{member.name} value '{member.value}' is not lowercase"
