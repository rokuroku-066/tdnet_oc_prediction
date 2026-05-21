from tdnet_oc_prediction.config.loader import load_config


def test_load_config_with_minimum_project_only(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("project: {name: custom_project}\n", encoding="utf-8")

    loaded = load_config(str(cfg))

    assert loaded.project.name == "custom_project"
    assert loaded.project.seed == 42
    assert loaded.project.timezone == "Asia/Tokyo"
    assert loaded.data == {}
    assert loaded.split == {}
    assert loaded.model == {}
    assert loaded.evaluation == {}
    assert loaded.simulation == {}
