from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Model:
    """
    A machine learning model for classifying the Iris dataset using the K-Nearest Neighbors (KNN) algorithm.

    Attributes:
        dataset (str): The name of the dataset used ('iris').
        architecture (str): The name of the model architecture ('KNN').
        features (list): List of feature names used in the dataset.
        labels (list): List of target labels for the dataset.
        _train_data (tuple): Tuple containing training data (X_train, y_train).
        _eval_data (tuple): Tuple containing evaluation data (X_test, y_test).
        model (KNeighborsClassifier): The trained KNN model.
        eval (float): Accuracy score of the model on the evaluation dataset.

    Methods:
        __call__(data): Predicts the labels for the given data.
        _init_data(test_size): Initializes and splits the Iris dataset into training and evaluation sets.
        _score(): Evaluates the model and returns the accuracy score.
        _train(test_size): Trains the KNN model using the Iris dataset.
    """

    def __init__(self, test_size=0.5):
        """
        Initializes the Model class by setting up the dataset, model architecture, and training the model.

        Args:
            test_size (float, optional): Proportion of the dataset to be used for testing. Default is 0.5.
        """
        self.dataset = "iris"
        self.architecture = "KNN"
        self._train(test_size)

    def __call__(self, data):
        """
        Predicts labels for the given input data using the trained model.

        Args:
            data (list of list of float/int): The input data records to predict. Each record should have the same number
            of features as the model was trained on.

        Yields:
            str: Predicted label for each input data record.

        Raises:
            ValueError: If the input data record is malformed (i.e., incorrect number of features or non-numeric values).
        """
        for record in data:
            if len(record) != len(self.labels) and not all(
                [isinstance(val, float) or isinstance(val, int) for val in record]
            ):
                raise ValueError(f"Malformed data record {record}")

        yield from (self.labels[label] for label in self.model.predict(data))

    def _init_data(self, test_size=0.5):
        """
        Loads the Iris dataset and splits it into training and evaluation sets.

        Args:
            test_size (float, optional): Proportion of the dataset to be used for testing. Default is 0.5.

        Attributes set:
            features (list): List of feature names from the Iris dataset.
            labels (list): List of target labels from the Iris dataset.
            _train_data (tuple): Training data (X_train, y_train).
            _eval_data (tuple): Evaluation data (X_test, y_test).
        """
        iris_dataset = load_iris()
        self.features = iris_dataset.feature_names
        self.labels = iris_dataset.target_names
        x_train, x_test, y_train, y_test = train_test_split(
            iris_dataset.data, iris_dataset.target, test_size=0.5
        )
        self._train_data = (x_train, y_train)
        self._eval_data = (x_test, y_test)

    def _score(self):
        """
        Evaluates the trained model using the evaluation dataset and calculates the accuracy score.

        Returns:
            float: Accuracy score of the model on the evaluation dataset.
        """
        preds = self.model.predict(self._eval_data[0])
        return accuracy_score(preds, self._eval_data[1])

    def _train(self, test_size=0.5):
        """
        Initializes the data and trains the KNN model on the training dataset.

        Args:
            test_size (float, optional): Proportion of the dataset to be used for testing. Default is 0.5.

        Sets the attributes:
            model (KNeighborsClassifier): Trained KNN model.
            eval (float): Accuracy score of the model on the evaluation dataset.
        """
        self._init_data()
        classifier = KNeighborsClassifier()
        classifier.fit(*self._train_data)
        self.model = classifier
        self.eval = self._score()
