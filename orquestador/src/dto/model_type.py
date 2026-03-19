from enum import Enum


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    LINEAR_REGRESSION = "linear_regression"
