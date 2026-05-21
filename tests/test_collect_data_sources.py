import pandas as pd
import pytest

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("collect_data", Path("scripts/collect_data.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
main = _mod.main
from tdnet_oc_prediction.data.stooq_client import StooqClient
from tdnet_oc_prediction.data.tdnet_client import TDnetPublicClient


def test_tdnet_public_client_parse(monkeypatch):
    client = TDnetPublicClient(sleep_sec=0)
    html = """
    <table><tr><td>15:00</td><td>7203</td><td>TOYOTA</td><td><a href='/docs/not_pdf'>XBRL</a><a href='/docs/a.pdf'>決算</a></td></tr></table>
    """

    class R:
        status_code = 200
        text = html
        apparent_encoding = "utf-8"

    class P:
        status_code = 404

    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        return R() if calls["n"] == 1 else P()

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr(client, "_extract_pdf_text", lambda url: "body")
    df = client.fetch("2024-01-01", "2024-01-01")
    assert len(df) == 1
    assert df.iloc[0]["stock_code"] == "7203"
    assert df.iloc[0]["body_text"] == "body"


def test_stooq_client_fetch(monkeypatch):
    client = StooqClient(sleep_sec=0)
    csv = "Date,Open,High,Low,Close,Volume\n2024-01-04,10,11,9,10.5,1000\n"

    class R:
        text = csv

        def raise_for_status(self):
            return None

    monkeypatch.setattr(client.session, "get", lambda url, timeout: R())
    df = client.fetch(["7203"], "2024-01-01", "2024-01-31")
    assert len(df) == 1
    assert list(df.columns) == ["stock_code", "date", "open", "high", "low", "close", "volume"]


def test_collect_data_raises_when_empty(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
project: {name: tdnet_oc_prediction}
data:
  disclosure_source: csv
  price_source: csv
  disclosure_path: tests/fixtures/empty_disclosures.csv
  price_path: tests/fixtures/empty_prices.csv
  start_date: '2024-01-01'
  end_date: '2024-01-31'
"""
    )
    with pytest.raises(RuntimeError):
        main(str(cfg))


def test_tdnet_public_client_without_pdf_extract(monkeypatch):
    client = TDnetPublicClient(sleep_sec=0, extract_pdf=False)
    html = """
    <table><tr><td>15:00</td><td>7203</td><td>TOYOTA</td><td><a href='/docs/a.pdf'>決算</a></td></tr></table>
    """

    class R:
        status_code = 200
        text = html
        apparent_encoding = "utf-8"
        apparent_encoding = "utf-8"

    class P:
        status_code = 404

    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        return R() if calls["n"] == 1 else P()

    monkeypatch.setattr(client.session, "get", fake_get)
    df = client.fetch("2024-01-01", "2024-01-01")
    assert len(df) == 1
    assert df.iloc[0]["body_text"] == ""


def test_collect_data_extends_price_period(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    out_d = tmp_path / "disclosures.csv"
    out_p = tmp_path / "prices.csv"
    cfg.write_text(
        f"""
project: {{name: tdnet_oc_prediction}}
data:
  disclosure_source: csv
  price_source: csv
  disclosure_path: {out_d}
  price_path: {out_p}
  start_date: '2024-01-01'
  end_date: '2024-01-31'
  price_extra_days: 10
"""
    )

    disclosures = pd.DataFrame([
        {
            "disclosure_id": "x",
            "disclosed_at": "2024-01-10T15:00:00+09:00",
            "stock_code": "7203",
            "title": "t",
            "body_text": "b",
        }
    ])
    prices = pd.DataFrame([
        {
            "stock_code": "7203",
            "date": "2024-02-09",
            "open": 1,
            "high": 1,
            "low": 1,
            "close": 1,
            "volume": 1,
        }
    ])

    class FakeDisclosureClient:
        def __init__(self, _path):
            pass

        def fetch(self, start, end):
            assert start == "2024-01-01"
            assert end == "2024-01-31"
            return disclosures

    class FakePriceClient:
        def __init__(self, _path):
            self.called = None

        def fetch(self, codes, start, end):
            assert codes == ["7203"]
            assert start == "2024-01-01"
            assert end == "2024-02-10"
            return prices

    monkeypatch.setattr(_mod, "DisclosureClient", FakeDisclosureClient)
    monkeypatch.setattr(_mod, "PriceClient", FakePriceClient)

    main(str(cfg))
    assert out_d.exists()
    assert out_p.exists()
