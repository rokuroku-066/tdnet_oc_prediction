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


def test_dataset_builder_output_schema_contains_text_column():
    disclosures = pd.DataFrame([
        {'disclosure_id':'1','stock_code':'7203','disclosure_date':'2024-01-04','title':'A','body_text':'a'},
    ])
    prices = pd.DataFrame([
        {'stock_code':'7203','date':'2024-01-05','open':100,'close':110},
    ])

    out = DatasetBuilder().build(disclosures, prices)

    assert "text" in out.columns


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


def test_dataset_builder_vectorized_columns_equivalent_to_legacy_logic():
    disclosures = pd.DataFrame([
        {'disclosure_id':'1','stock_code':'7203','disclosure_date':'2024-01-04','title':'A','body_text':'a'},
        {'disclosure_id':'2','stock_code':'7203','disclosure_date':'2024-01-04','title':'B','body_text':'b'},
        {'disclosure_id':'3','stock_code':'6758','disclosure_date':'2024-01-04','title':'C','body_text':'c'},
        {'disclosure_id':'4','stock_code':'9984','disclosure_date':'2024-01-04','title':'D','body_text':'d'},
    ])
    prices = pd.DataFrame([
        {'stock_code':'7203','date':'2024-01-05','open':100,'close':110},
        {'stock_code':'6758','date':'2024-01-05','open':200,'close':150},
        {'stock_code':'9984','date':'2024-01-05','open':300,'close':300},
    ])

    out = DatasetBuilder().build(disclosures, prices)

    d = disclosures.copy()
    d['disclosure_date'] = pd.to_datetime(d['disclosure_date']).dt.normalize()
    d['text_piece_legacy'] = d.apply(lambda r: f"[{r['title']}]\n{r['body_text']}", axis=1)
    expected_text = d.groupby(['stock_code','disclosure_date'], as_index=False).agg(
        text=('text_piece_legacy', '\n\n'.join),
        num_disclosures=('disclosure_id', 'count')
    )
    expected_text['target_date'] = pd.to_datetime('2024-01-05')

    p = prices.copy()
    p['date'] = pd.to_datetime(p['date']).dt.normalize()
    expected = expected_text.merge(
        p[['stock_code','date','open','close']],
        left_on=['stock_code','target_date'],
        right_on=['stock_code','date'],
        how='left'
    ).rename(columns={'open':'target_open','close':'target_close'}).drop(columns=['date'])
    expected['y'] = expected.apply(
        lambda r: 1 if r['target_close'] > r['target_open'] else (0 if r['target_close'] < r['target_open'] else None),
        axis=1,
    )
    expected = expected.dropna(subset=['target_date', 'target_open', 'target_close', 'y']).copy()
    expected['y'] = expected['y'].astype(int)
    expected['sample_id'] = expected.apply(lambda r: f"{r['stock_code']}_{r['disclosure_date'].date()}", axis=1)
    expected = expected[[
        'sample_id','stock_code','disclosure_date','target_date','text','target_open','target_close','y','num_disclosures'
    ]].sort_values(['stock_code']).reset_index(drop=True)

    actual = out.sort_values(['stock_code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
