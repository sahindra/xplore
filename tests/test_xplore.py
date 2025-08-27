import pandas as pd
from xplore import explore_feature

def test_numeric_feature_runs():
    df = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "pts": [10, 20, 15, 25]
    })
    summary = explore_feature(df, "target", "pts", show=False)
    assert "mean" in summary.columns

def test_categorical_feature_runs():
    df = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "team": ["A", "B", "A", "B"]
    })
    counts = explore_feature(df, "target", "team", show=False)
    assert "count" in counts.columns
