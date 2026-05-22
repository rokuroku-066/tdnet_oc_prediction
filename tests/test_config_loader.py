import pytest

from tdnet_oc_prediction.config.loader import load_config


def test_load_config_with_minimum_project_only(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("project: {name: custom_project}\n", encoding="utf-8")

    loaded = load_config(str(cfg))

    assert loaded.project.name == "custom_project"
    assert loaded.project.seed == 42
    assert loaded.project.timezone == "Asia/Tokyo"
    assert loaded.data.disclosure_source == "csv"
    assert loaded.model.name == "tfidf_logreg"
    assert loaded.evaluation.threshold == 0.5
    assert loaded.simulation.threshold_long == 0.55
    assert loaded.split is None


def test_load_config_fails_for_invalid_key(tmp_path):
    cfg = tmp_path / "invalid_key.yaml"
    cfg.write_text("model:\n  name: tfidf_logreg\n  unknown_key: x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid config section 'model'"):
        load_config(str(cfg))


def test_load_config_fails_for_invalid_type(tmp_path):
    cfg = tmp_path / "invalid_type.yaml"
    cfg.write_text("model:\n  threshold: high\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid config section 'model'"):
        load_config(str(cfg))


def test_load_config_fails_for_missing_required_field(tmp_path):
    cfg = tmp_path / "missing_required.yaml"
    cfg.write_text("split:\n  method: time_series\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid config section 'split'"):
        load_config(str(cfg))
