import json

from click.testing import CliRunner

from harness.app.cli.main import cli
from harness.ml.runners.artifacts import save_artifact


def test_inspect_bundle_does_not_deserialize_payload(tmp_path):
    path = tmp_path / "model.bundle"
    save_artifact(
        path,
        {"executable": "payload"},
        {
            "task": {"task_type": "binary"},
            "models": [],
            "training_features": [],
            "ensemble_columns": [],
            "fingerprints": {"training_data": "abc"},
            "output": {"kind": "scalar"},
        },
    )

    result = CliRunner().invoke(cli, ["inspect-bundle", str(path)])

    assert result.exit_code == 0, result.output
    manifest = json.loads(result.output)
    assert manifest["format_version"] == 1
    assert manifest["payload"]["sha256"]
