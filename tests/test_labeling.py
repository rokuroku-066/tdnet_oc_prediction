from tdnet_oc_prediction.data.dataset_builder import make_label

def test_labeling():
    assert make_label(100, 101) == 1
    assert make_label(100, 99) == 0
    assert make_label(100, 100) is None
