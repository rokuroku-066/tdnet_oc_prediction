from __future__ import annotations

import pytest

pytest.importorskip("datasets")
from tdnet_oc_prediction.models.transformer_classifier import TransformerClassifier


def test_load_calls_from_pretrained_once_each_and_sets_attributes(monkeypatch):
    tokenizer_calls: list[tuple[str, dict]] = []
    model_calls: list[tuple[str, dict]] = []

    class DummyTokenizer:
        pass

    class DummyModel:
        pass

    def fake_tokenizer_from_pretrained(path, **kwargs):
        tokenizer_calls.append((path, kwargs))
        return DummyTokenizer()

    def fake_model_from_pretrained(path, **kwargs):
        model_calls.append((path, kwargs))
        return DummyModel()

    monkeypatch.setattr(
        "tdnet_oc_prediction.models.transformer_classifier.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "tdnet_oc_prediction.models.transformer_classifier.AutoModelForSequenceClassification.from_pretrained",
        fake_model_from_pretrained,
    )

    model = TransformerClassifier.load("/tmp/fake-model-dir")

    assert len(tokenizer_calls) == 1
    assert len(model_calls) == 1
    assert tokenizer_calls[0] == ("/tmp/fake-model-dir", {})
    assert model_calls[0] == ("/tmp/fake-model-dir", {})

    assert model.config["pretrained_model_name"] == "/tmp/fake-model-dir"
    assert model.max_length == 512
    assert model.batch_size == 16
    assert isinstance(model.tokenizer, DummyTokenizer)
    assert isinstance(model.model, DummyModel)


def test_init_still_loads_pretrained_components(monkeypatch):
    tokenizer_calls: list[tuple[str, dict]] = []
    model_calls: list[tuple[str, dict]] = []

    class DummyTokenizer:
        pass

    class DummyModel:
        pass

    def fake_tokenizer_from_pretrained(path, **kwargs):
        tokenizer_calls.append((path, kwargs))
        return DummyTokenizer()

    def fake_model_from_pretrained(path, **kwargs):
        model_calls.append((path, kwargs))
        return DummyModel()

    monkeypatch.setattr(
        "tdnet_oc_prediction.models.transformer_classifier.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "tdnet_oc_prediction.models.transformer_classifier.AutoModelForSequenceClassification.from_pretrained",
        fake_model_from_pretrained,
    )

    model = TransformerClassifier(
        {
            "pretrained_model_name": "dummy-pretrained",
            "max_length": 128,
            "batch_size": 4,
        }
    )

    assert tokenizer_calls == [("dummy-pretrained", {})]
    assert model_calls == [("dummy-pretrained", {"num_labels": 2})]
    assert model.max_length == 128
    assert model.batch_size == 4
    assert isinstance(model.tokenizer, DummyTokenizer)
    assert isinstance(model.model, DummyModel)
