import pandas as pd
from tdnet_oc_prediction.data.dataset_builder import DatasetBuilder

def test_dataset_builder_aggregate_and_target():
    d = pd.DataFrame([
        {'disclosure_id':'1','stock_code':'7203','disclosure_date':'2024-01-04','title':'A','body_text':'a'},
        {'disclosure_id':'2','stock_code':'7203','disclosure_date':'2024-01-04','title':'B','body_text':'b'},
    ])
    p = pd.DataFrame([{'stock_code':'7203','date':'2024-01-05','open':100,'close':110}])
    out = DatasetBuilder().build(d,p)
    assert len(out)==1 and out.iloc[0]['num_disclosures']==2 and out.iloc[0]['y']==1


def test_dataset_builder_calendar_priority_for_target_date():
    d = pd.DataFrame([
        {'disclosure_id':'1','stock_code':'7203','disclosure_date':'2024-01-04','title':'A','body_text':'a'},
    ])
    p = pd.DataFrame([
        {'stock_code':'7203','date':'2024-01-05','open':100,'close':101},
        {'stock_code':'7203','date':'2024-01-08','open':200,'close':210},
    ])
    calendar = pd.DataFrame([
        {'date':'2024-01-08'},
    ])

    out = DatasetBuilder().build(d, p, calendar=calendar)

    assert len(out) == 1
    assert str(out.iloc[0]['target_date'].date()) == '2024-01-08'
    assert out.iloc[0]['target_open'] == 200


def test_dataset_builder_calendar_validation_missing_date_column():
    d = pd.DataFrame([
        {'disclosure_id':'1','stock_code':'7203','disclosure_date':'2024-01-04','title':'A','body_text':'a'},
    ])
    p = pd.DataFrame([{'stock_code':'7203','date':'2024-01-05','open':100,'close':101}])
    calendar = pd.DataFrame([{'business_day':'2024-01-08'}])

    try:
        DatasetBuilder().build(d, p, calendar=calendar)
        assert False, "Expected ValueError for missing calendar date column"
    except ValueError as e:
        assert "calendar is missing required columns" in str(e)
