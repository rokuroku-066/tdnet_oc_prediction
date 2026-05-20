import pandas as pd
import pytest
from tdnet_oc_prediction.data.dataset_builder import DatasetBuilder, TimeSeriesSplitter

def test_dataset_builder_aggregate_and_target():
    d = pd.DataFrame([
        {'disclosure_id':'1','stock_code':'7203','disclosure_date':'2024-01-04','title':'A','body_text':'a'},
        {'disclosure_id':'2','stock_code':'7203','disclosure_date':'2024-01-04','title':'B','body_text':'b'},
    ])
    p = pd.DataFrame([{'stock_code':'7203','date':'2024-01-05','open':100,'close':110}])
    out = DatasetBuilder().build(d,p)
    assert len(out)==1 and out.iloc[0]['num_disclosures']==2 and out.iloc[0]['y']==1


def test_time_series_splitter_raises_on_overlapped_periods():
    dataset = pd.DataFrame([
        {"disclosure_date": "2024-01-05"},
        {"disclosure_date": "2024-01-10"},
        {"disclosure_date": "2024-01-15"},
    ])
    splitter = TimeSeriesSplitter({
        "train_end": "2024-01-10",
        "valid_start": "2024-01-10",
        "valid_end": "2024-01-12",
        "test_start": "2024-01-13",
        "test_end": "2024-01-20",
    })

    with pytest.raises(ValueError):
        splitter.split(dataset)
