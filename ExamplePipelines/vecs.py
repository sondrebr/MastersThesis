"""Examples of vectorization classes for use with BenchTDA."""

from typing import Literal

import numpy as np

from gudhi.representations import Landscape, PersistenceImage
from sklearn.neural_network import MLPClassifier, MLPRegressor

from BenchTDA import ModelBase, VectorizationBase


DEFAULT_HPARAM_RANGES: dict[tuple[str, Literal["discrete", "continuous"]], list] = {
    ("hidden_layers", "discrete"): [
        tuple([32 for _ in range(i+1)]) for i in range(5)  # (32, ), (32, 32, ), etc.
    ],
    ("learning_rate_exp", "continuous"): [-4, 0],  # Unpack as 10**y
}


"""
    MODEL CLASS EXAMPLES
"""


class DefaultClassifier(ModelBase):
    """Simple classifier class for use with vectorization classes."""

    def __init__(self, hparams: dict, seed: int):
        """Unpack hyperparameters and initialize the model w/ seed.

        Args:
            hparams (dict): Hyperparameters, must use the same names as DEFAULT_HPARAM_RANGES.
            seed (int): Used to set the random_state of the model.
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hparams["hidden_layers"],
            learning_rate_init=10**hparams["learning_rate_exp"],
            random_state=seed,
            early_stopping=True
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model using data X and labels y.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Labels for training data X.
        """
        self.model = self.model.fit(X, y)

    def predict(self, X):
        """Use the trained model to predict y based on X.

        Args:
            X: Input data of shape (n_samples, n_features) to predict labels for.

        Returns:
            np.ndarray: Predicted labels of input data X.
        """
        return np.array(self.model.predict(X))


class DefaultRegressor(ModelBase):
    """Simple classifier class for use with vectorization classes."""

    def __init__(self, hparams: dict, seed: int):
        """Unpack hyperparameters and initialize the model w/ seed.

        Args:
            hparams (dict): Hyperparameters, must use the same names as DEFAULT_HPARAM_RANGES.
            seed (int): Used to set the random_state of the model.
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hparams["hidden_layers"],
            learning_rate_init=10**hparams["learning_rate_exp"],
            random_state=seed,
            early_stopping=True
        )

    def fit(self, X, y):
        """Train the model using data X and labels y.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Labels for training data X.
        """
        self.model = self.model.fit(X, y)

    def predict(self, X):
        """Use the trained model to predict y based on X.

        Args:
            X: Input data of shape (n_samples, n_features) to predict values for.

        Returns:
            np.ndarray: Predicted values of input data X.
        """
        return np.array(self.model.predict(X))


"""
    VECTORIZATION CLASS EXAMPLES
"""


class ConcatenationBaseline(VectorizationBase):
    """Non-TDA baseline with concatenated points for comparison with TDA pipelines."""

    vec_parameter_ranges = {}
    hyperparameter_ranges = DEFAULT_HPARAM_RANGES

    classifier = DefaultClassifier
    regressor = DefaultRegressor

    def __init__(self, seed: int):
        """Only stores seed, in case it is needed."""
        self.seed = seed

    def fit(self, X, vec_params):
        """Do nothing."""
        pass

    def transform(self, X):
        """Concatenate data X for each sample (point cloud, etc.).

        Args:
            X (np.ndarray): Dataset X with shape (n_samples, n_points, n_features).

        Returns:
            np.ndarray: Concatenated points per point cloud
        """
        return X.reshape((X.shape[0], X.shape[1]*X.shape[2]))


class MeanVarBaseline(VectorizationBase):
    """Mean/var.-based non-TDA baseline for comparison with TDA pipelines."""

    vec_parameter_ranges = {}
    hyperparameter_ranges = DEFAULT_HPARAM_RANGES

    classifier = DefaultClassifier
    regressor = DefaultRegressor

    def __init__(self, seed: int):
        """Only stores seed, in case it is needed."""
        self.seed = seed

    def fit(self, X, vec_params):
        """Do nothing."""
        pass

    def transform(self, X):
        """Convert data X to [mean, var] of each feature for each sample (point cloud, etc.).

        Args:
            X (np.ndarray): Dataset X with shape (n_samples, n_points, n_features).

        Returns:
            np.ndarray: Array with [mean, var] of each feature for each sample
        """
        mean = np.mean(X, axis=1)
        var = np.var(X, axis=1)

        out = np.hstack((mean, var))
        return out


class PLandscape(VectorizationBase):
    """Example of a vectorization class using gudhi persistence landscapes."""

    # Define vectorization parameters w/ ranges
    vec_parameter_ranges = {
        ("nl", "discrete"): list(range(1, 6)),
        ("res", "discrete"): [2**i for i in range(5, 10)],  # Min. 32, max. 512
    }
    hyperparameter_ranges = DEFAULT_HPARAM_RANGES

    classifier = DefaultClassifier
    regressor = DefaultRegressor

    def __init__(self, seed: int):
        """Store seed in case it is needed later."""
        self.seed = seed

    def fit(self, X, vec_params):
        """Instantiate a gudhi Landscape and fit to the training persistence diagrams in X.

        Args:
            X: List of training persistence diagrams.
            vec_params: Vectorization parameters dict {"nl": int, "res": int}.
        """
        self.landscaper = Landscape(
            num_landscapes=vec_params["nl"],
            resolution=vec_params["res"]
        ).fit(X, [])

    def transform(self, X):
        """Convert the persistence diagrams in X to persistence landscapes.

        Args:
            X: List of persistence diagrams to convert.

        Returns:
            np.ndarray: Persistence landscapes, shape (n_diagrams, nl * res).
        """
        return self.landscaper.transform(X)


class PImage(VectorizationBase):
    """Example of a vectorization class using gudhi persistence landscapes."""

    vec_parameter_ranges = {
        ("res", "discrete"): [4, 8, 10, 12, 16],  # Unpack [y, y]
        ("bw_exp", "continuous"): [-3, 3],  # Unpack as 2**y
        ("weight_exp", "continuous"): [-1, 2]  # Unpack as lambda x: x[1]**max(0, y)
    }
    hyperparameter_ranges = DEFAULT_HPARAM_RANGES

    classifier = DefaultClassifier
    regressor = DefaultRegressor

    def __init__(self, seed=42):
        """Store seed in case it is needed later."""
        self.seed = seed

    def fit(self, X: list, vec_params):
        """Instantiate a gudhi Landscape and fit to the training persistence diagrams in X.

        Args:
            X: List of training persistence diagrams.
            vec_params: Vectorization parameters dict {"res": int, "bw_exp": float, "weight_exp": float}.
        """
        self.imager = PersistenceImage(
            resolution=[vec_params["res"], vec_params["res"]],
            bandwidth=2**vec_params["bw_exp"],
            weight=lambda x: x[1]**max(0, vec_params["weight_exp"]),
        ).fit(X)

    def transform(self, X):
        """Convert the persistence diagrams in X to persistence images.

        Args:
            X: List of persistence diagrams to convert.

        Returns:
            np.ndarray: Persistence images, shape (n_diagrams, res * res)
        """
        return self.imager.transform(X)
