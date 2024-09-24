import json
import os

from .train import Model


def main():
    """
    Main function that initializes a machine learning model, loads data from the environment,
    makes predictions using the model, and saves the results in a JSON file.

    Raises:
        ValueError: If no data is provided in the 'DATA' environment variable.

    The output JSON file contains a list of records, each with the following fields:
        - dataset: The name of the dataset used for training the model (e.g., 'iris').
        - architecture: The architecture of the model used (e.g., 'KNN').
        - features: The evaluation score (accuracy) of the model.
        - data: The input data record.
        - label: The predicted label for the data record.
    """
    m = Model()
    data = os.getenv("DATA")
    if not data:
        raise ValueError("No data provided")

    data = json.loads(data)
    records = [
        {
            "dataset": m.dataset,
            "architecture": m.architecture,
            "features": m.eval,
            "data": record,
            "label": label,
        }
        for record, label in zip(data, m(data))
    ]

    json.dump(records, open("out.json", "w"))
