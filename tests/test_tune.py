import pandas as pd
from sklearn.datasets import make_classification

# OPTION A: if tune.py is inside src/  ->  src/tune.py
from src.model.tune import tune_model

# OPTION B: if tune.py is at project root (E:/CHURN/tune.py), use this instead:
# from tune import tune_model


def test_tune_model_returns_params():
    # small synthetic dataset
    X_arr, y_arr = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=3,
        random_state=0,
    )
    X = pd.DataFrame(X_arr)
    y = pd.Series(y_arr)

    best_params = tune_model(X, y)

    # Check that we got a dict
    assert isinstance(best_params, dict)
    # And that some expected hyperparameters are present
    assert "n_estimators" in best_params
    assert "learning_rate" in best_params
    assert "max_depth" in best_params
