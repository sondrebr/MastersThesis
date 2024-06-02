from abc import abstractmethod
from typing import Any, Literal
from numpy import ndarray

parameter_range_type = dict[tuple[str, Literal["continuous", "discrete"]], list]


class ModelBase:
    """Base class for the model classes to be included in the vectorization class."""

    model: Any

    @abstractmethod
    def __init__(self, params: dict, seed: int):
        """Initialize the model.

        Args:
            params (dict): Dictionary containing the hyperparameters to be used by the model.
            seed (int): The seed to be used for the model.

        Raises:
            NotImplementedError: __init__ must be implemented by the model class.
        """
        raise NotImplementedError()

    # Train self.model with training data X, training labels y
    # For example, assuming an sklearn model:
    # def fit(self, X, y):
    #  self.model = self.model.fit(X, y)
    @abstractmethod
    def fit(self, X: ndarray, y: ndarray):
        """Fit the model to the data.

        Args:
            X (ndarray): Training data.
            y (ndarray): Training labels.

        Raises:
            NotImplementedError: fit must be implemented by the model class.
        """
        raise NotImplementedError()

    # Standard sklearn-like predict - data in, prediction out
    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        """Use the model to predict based on the data.

        Args:
            X (ndarray): Input data.

        Raises:
            NotImplementedError: predict must be implemented by the model class.

        Returns:
            ndarray: The predictions.
        """
        raise NotImplementedError()


class VectorizationBase:
    """Base class for the vectorization classes."""

    # Defines vectorization parameter ranges in the form
    # (name: str, "continuous"): [start, stop]
    # or
    # (name: str, "discrete"): [v_1, ..., v_n]
    vec_parameter_ranges: parameter_range_type | None

    # Defines hyperparameter ranges in the form
    # (name: str, "continuous"): [start, stop]
    # or
    # (name: str, "discrete"): [p_1, ..., p_n]
    hyperparameter_ranges: parameter_range_type | None

    # Class itself, not instance
    classifier: type[ModelBase] | Any
    regressor: type[ModelBase] | Any

    @abstractmethod
    def __init__(self, seed: int):
        """Initialize the vectorization class.

        Args:
            seed (int): Seed to use for vectorization, in case it involves randomness.

        Raises:
            NotImplementedError: __init__ must be implemented in the vectorization class.
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, X: ndarray, vec_params: dict[str, Any]) -> None:
        """Initialize vectorization based on training data X and vec_params.

        Args:
            X (np.ndarray): Training data
            vec_params (dict): Parameters for vectorization
            seed (int): Seed in case vectorization involves randomization

        Raises:
            NotImplementedError: fit must be implemented in the vectorization class.
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, X: ndarray) -> ndarray:
        """Vectorize the data.

        Args:
            X (ndarray): The data to be vectorized.

        Raises:
            NotImplementedError: transform must be implemented in the vectorization class.

        Returns:
            ndarray: The vectorized data.
        """
        raise NotImplementedError()
