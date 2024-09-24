# tests/test_ml_app.py
from ml_app.train import Model


def test_model_initialization():
    """
    Test the initialization of the Model class.

    Ensures that the model attributes are correctly set during initialization and
    that the evaluation accuracy score is non-negative.
    """
    model = Model()
    assert model.dataset == "iris"
    assert model.architecture == "KNN"
    assert model.eval >= 0  # Check that accuracy score is >= 0


def test_inference():
    """
    Test the inference functionality of the Model class.

    Verifies that the model correctly predicts the class label for a given input data point.

    Raises:
        AssertionError: If the prediction does not match the expected class label.
    """
    model = Model()
    test_data = [
        [5.1, 3.5, 1.4, 0.2]
    ]  # Example data point that should map to a specific class
    predictions = list(model(test_data))
    assert predictions == ["setosa"], "The prediction should match the expected class"
