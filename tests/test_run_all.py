import json
import importlib.util
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location("run_all", Path("scripts/run_all.py"))
run_all = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(run_all)


def test_run_capture_json_run_id_success(monkeypatch):
    def _mock_run(cmd, check, capture_output, text):
        assert "--quiet-json" in cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps({"run_id": "abc123"}), stderr="")

    monkeypatch.setattr(subprocess, "run", _mock_run)
    assert run_all.run_capture_json_run_id(["python", "scripts/train.py", "--quiet-json"], "train") == "abc123"


def test_run_capture_json_run_id_invalid_json(monkeypatch):
    def _mock_run(cmd, check, capture_output, text):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="not-json", stderr="")

    monkeypatch.setattr(subprocess, "run", _mock_run)
    with pytest.raises(RuntimeError, match="stdout is not valid JSON"):
        run_all.run_capture_json_run_id(["python", "scripts/train.py", "--quiet-json"], "train")
