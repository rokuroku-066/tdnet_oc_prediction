import pandas as pd
from tdnet_oc_prediction.simulation.backtester import Backtester

def test_simulation_returns():
    df = pd.DataFrame([
        {'sample_id':'a','stock_code':'x','target_date':'2024-01-01','target_open':100,'target_close':110,'y_true':1,'y_proba':0.9,'signal':1},
        {'sample_id':'b','stock_code':'x','target_date':'2024-01-02','target_open':100,'target_close':90,'y_true':0,'y_proba':0.1,'signal':-1},
    ])
    t = Backtester().run(df, transaction_cost_bps=10)
    assert round(t.iloc[0]['gross_return'],4)==0.1
    assert round(t.iloc[1]['gross_return'],4)==0.1
    assert round(t.iloc[0]['net_return'],4)==0.099
