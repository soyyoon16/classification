# **Machine Learning Model Deployment for Iris Classification**

## **Overview**

This project demonstrates an end-to-end Machine Learning Operations (MLOps) pipeline using GitHub Actions for continuous integration and continuous deployment (CI/CD). We use the Iris dataset, a classic machine learning classification problem, to build, train, and deploy a machine learning model. This project focuses on understanding the MLOps infrastructure and inference API design rather than the complexity of the model itself.

## **Key Features**

- **Model Training and Inference**: We utilize the Iris dataset to train a Random Forest Classifier that predicts the species of an iris flower based on petal and sepal measurements.
- **CI/CD Integration**: The project is integrated with GitHub Actions to automate tasks such as model training, testing, deployment, and documentation generation. Every push or pull request triggers the CI/CD pipeline.
- **Automated Testing**: Unit tests are implemented using `pytest` to ensure the model and code functionality are valid and reliable.
- **Documentation**: Comprehensive documentation is generated using Sphinx and hosted on ReadTheDocs.
- **Model Serialization**: The trained model is serialized using `pickle` and saved as an artifact in the CI/CD pipeline.

## **Technologies and Tools**

- **Machine Learning**: `scikit-learn`
- **CI/CD**: GitHub Actions
- **Documentation**: Sphinx with `autoapi` and `sphinx_rtd_theme`
- **Python Packaging**: Poetry
- **Visualization**: `plotly` for interactive visualizations
- **Testing**: `pytest` for unit tests

## **Project Structure**
classification/
├── docs/                           # Sphinx documentation files
├── ml_app/                         # Core project package
│   ├── __init__.py
│   ├── train.py                    # Model training script
│   ├── inference.py                # Model inference script
│   └── visualize.py                # Visualization script
├── tests/                          # Unit test files
│   └── test_ml_app.py
├── README.md                       # Project description and instructions
├── pyproject.toml                  # Poetry configuration file
└── .github/workflows/ml_app-ci.yml # GitHub Actions workflow file


