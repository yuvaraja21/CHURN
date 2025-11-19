# unit + integration tests for build_features/_map_binary_series
import json
import pandas as pd
import pandas.testing as pdt
import pytest
from src.features.build_features import build_features, _map_binary_series

# ---------- Unit tests for helper ----------
def test_map_binary_series_basic_yes_no():
    s = pd.Series(["Yes", "No", "Yes", None])
    mapped = _map_binary_series(s)
    # Expect dtype to be pandas nullable Int64 (or int-like)
    assert str(mapped.dtype).lower().startswith("int")
    # Values
    assert mapped.iloc[0] == 1
    assert mapped.iloc[1] == 0
    # None preserved as <NA>
    assert pd.isna(mapped.iloc[3])

@pytest.mark.parametrize("vals,expected", [
    (["Male","Female","Male"], [1,0,1]),
    (["No","Yes","No"], [0,1,0]),
])
def test_map_binary_series_parametrized(vals, expected):
    s = pd.Series(vals)
    mapped = _map_binary_series(s)
    assert list(mapped.dropna().astype(int)) == expected

# ---------- Integration test for build_features ----------
def make_small_telco_df():
    return pd.DataFrame({
        "customerID": ["A","B","C"],
        "gender": ["Male","Female","Female"],
        "Partner": ["Yes","No","No"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "SeniorCitizen": [0, 1, 0],
        "TotalCharges": ["100.0", "200.5", ""],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "Churn": ["No", "Yes", "No"]
    })

def test_build_features_integration(tmp_path):
    df = make_small_telco_df()
    processed, encoders = build_features(df, target_col="Churn", mode="train")
    assert "Churn" in df.columns
    for col in ["gender", "Partner"]:
        assert processed[col].dtype == "int64" or str(processed[col].dtype).startswith("Int")
        assert set(processed[col].dropna().astype(int).unique()).issubset({0,1})
    assert "TotalCharges" in processed.columns
    assert processed["TotalCharges"].dtype in ("float64", "int64")
    dummies = [c for c in processed.columns if c.startswith("InternetService_")]
    assert len(dummies) >= 1
    for col in ["gender", "Partner"]:
        assert pd.api.types.is_integer_dtype(processed[col]) and not pd.api.types.is_dtype_equal(processed[col].dtype, "Int64")
    enc_path = tmp_path / "encoders.json"
    with open(enc_path, "w") as f:
        json.dump(encoders, f)
    with open(enc_path) as f:
        loaded = json.load(f)
    assert isinstance(loaded, dict)

# ---------- Edge case tests ----------
def test_unseen_category_handled_gracefully():
    df = pd.DataFrame({
        "gender": ["Male"],
        "Partner": ["Yes"],
        "InternetService": ["UnknownCategory"],
        "TotalCharges": ["0"],
        "MonthlyCharges": [10.0],
        "Churn": ["No"]
    })
    processed, encoders = build_features(df, target_col="Churn", mode="serve")
    assert isinstance(processed, pd.DataFrame)
