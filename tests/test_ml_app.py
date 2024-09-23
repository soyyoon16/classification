# tests/test_ml_app.py
from ml_app.train import Model


def test_model_initialization():
    model = Model()
    assert model.dataset == "iris"
    assert model.architecture == "KNN"
    assert model.eval >= 0  # Check that accuracy score is >= 0


def test_inference():
    model = Model()
    test_data = [
        [5.1, 3.5, 1.4, 0.2]
    ]  # Example data point that should map to a specific class
    predictions = list(model(test_data))
    assert predictions == ["setosa"], "The prediction should match the expected class"
