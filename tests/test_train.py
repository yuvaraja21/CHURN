# tests/test_train.py
from src.model.train import train_model
import pandas as pd

def test_train_creates_model(tmp_path):
    # make tiny dataset fixture
    df = pd.DataFrame({
        "A": [1,2,3,4],
        "B": [5,6,7,8],
        "Churn": [0,1,0,1]
    })
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    # pass tmp_path / "out" as model_out_dir (Path object)
    meta, model_fp = train(str(csv), model_out_dir=tmp_path / "model_out")

    assert "accuracy" in meta
    assert "recall" in meta
    assert model_fp is not None
    from pathlib import Path
    assert Path(model_fp).exists()
