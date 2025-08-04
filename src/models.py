import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseModel(ABC):
    """
    Abstract base class for all regression models.
    Implements common functionality for mini-batch SGD training.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 1000,
        tolerance: float = 1e-6,
    ):
        """
        Initialize the model.

        Args:
            learning_rate: Learning rate for gradient descent
            batch_size: Size of mini-batches for SGD
            max_epochs: Maximum number of training epochs
            tolerance: Convergence tolerance for early stopping
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.weights: Optional[np.ndarray] = None
        self.training_history = {"loss": [], "accuracy": []}

    def initialize_weights(self, n_features: int) -> None:
        """
        Initialize model weights.

        Args:
            n_features: Number of features (including bias)
        """
        # Xavier initialization
        self.weights = np.random.normal(0, np.sqrt(2.0 / n_features), n_features)

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for input samples.

        Args:
            X: Input features

        Returns:
            Predicted probabilities
        """
        pass

    @abstractmethod
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss for given predictions.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Loss value
        """
        pass

    @abstractmethod
    def compute_gradient(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to weights.

        Args:
            X: Input features
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Gradient vector
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions (0 or 1).

        Args:
            X: Input features

        Returns:
            Binary predictions
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Accuracy score
        """
        return np.mean(y_true == y_pred)

    def create_mini_batches(self, X: np.ndarray, y: np.ndarray) -> list:
        """
        Create mini-batches for SGD.

        Args:
            X: Input features
            y: Target labels

        Returns:
            List of (X_batch, y_batch) tuples
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)

        mini_batches = []
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            mini_batches.append((X[batch_indices], y[batch_indices]))

        return mini_batches

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model using mini-batch SGD.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history
        """
        if self.weights is None:
            self.initialize_weights(X_train.shape[1])

        prev_loss = float("inf")

        for epoch in range(self.max_epochs):
            # Create mini-batches
            mini_batches = self.create_mini_batches(X_train, y_train)

            epoch_loss = 0.0
            for X_batch, y_batch in mini_batches:
                # Forward pass
                y_pred = self.predict_proba(X_batch)

                # Compute loss and gradient
                batch_loss = self.compute_loss(y_batch, y_pred)
                gradient = self.compute_gradient(X_batch, y_batch, y_pred)

                # Update weights (weights is guaranteed to be not None here)
                if self.weights is not None:
                    self.weights -= self.learning_rate * gradient
                epoch_loss += batch_loss

            # Average loss over all batches
            epoch_loss /= len(mini_batches)

            # Calculate training accuracy
            train_pred = self.predict(X_train)
            train_acc = self.accuracy(y_train, train_pred)

            # Store training history
            self.training_history["loss"].append(epoch_loss)
            self.training_history["accuracy"].append(train_acc)

            # Validation metrics
            val_acc = None
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_acc = self.accuracy(y_val, val_pred)
                if "val_accuracy" not in self.training_history:
                    self.training_history["val_accuracy"] = []
                self.training_history["val_accuracy"].append(val_acc)

            # Early stopping check
            if abs(prev_loss - epoch_loss) < self.tolerance:
                logger.info(f"Converged at epoch {epoch + 1}")
                break

            prev_loss = epoch_loss

            # Log progress
            if (epoch + 1) % 100 == 0:
                val_info = f", Val Acc: {val_acc:.4f}" if val_acc is not None else ""
                logger.info(
                    f"Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}{val_info}"
                )

        return self.training_history


class LinearRegression(BaseModel):
    """
    Linear Regression model for binary classification.
    Uses Mean Squared Error loss function.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using linear function.

        Args:
            X: Input features

        Returns:
            Predicted probabilities (clipped to [0, 1])
        """
        linear_output = X @ self.weights
        # Clip to [0, 1] range for probability interpretation
        return np.clip(linear_output, 0, 1)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            MSE loss
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def compute_gradient(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of MSE loss.

        Args:
            X: Input features
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Gradient vector
        """
        n_samples = X.shape[0]
        return -2 * X.T @ (y_true - y_pred) / n_samples


class LogisticRegression(BaseModel):
    """
    Logistic Regression model for binary classification.
    Uses Cross-entropy loss function.
    """

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with numerical stability.

        Args:
            z: Input values

        Returns:
            Sigmoid output
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using logistic function.

        Args:
            X: Input features

        Returns:
            Predicted probabilities
        """
        linear_output = X @ self.weights
        return self.sigmoid(linear_output)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Cross-entropy loss.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Cross-entropy loss
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return float(
            -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        )

    def compute_gradient(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss.

        Args:
            X: Input features
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Gradient vector
        """
        n_samples = X.shape[0]
        return X.T @ (y_pred - y_true) / n_samples


class SoftmaxRegression(BaseModel):
    """
    Softmax Regression model for binary classification.
    Uses Cross-entropy loss function with softmax activation.
    Note: For binary classification, this is equivalent to logistic regression
    but implemented with 2 output classes for educational purposes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = 2  # Binary classification

    def initialize_weights(self, n_features: int) -> None:
        """
        Initialize weights for softmax regression.

        Args:
            n_features: Number of features (including bias)
        """
        # Initialize weights for 2 classes
        self.weights = np.random.normal(
            0, np.sqrt(2.0 / n_features), (n_features, self.n_classes)
        )

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function with numerical stability.

        Args:
            z: Input values (n_samples, n_classes)

        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using softmax function.

        Args:
            X: Input features

        Returns:
            Predicted probabilities for class 1 (win)
        """
        linear_output = X @ self.weights  # (n_samples, 2)
        probabilities = self.softmax(linear_output)
        # Return probability of class 1 (win)
        return probabilities[:, 1]

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Cross-entropy loss for softmax.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities for class 1

        Returns:
            Cross-entropy loss
        """
        # For binary classification, we can compute cross-entropy directly
        # from the probability of class 1
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return float(
            -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        )

    def compute_gradient(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss for softmax.

        Args:
            X: Input features
            y_true: True labels
            y_pred: Predicted probabilities for class 1

        Returns:
            Gradient matrix
        """
        n_samples = X.shape[0]

        # Convert to one-hot encoding
        y_true_onehot = np.column_stack([1 - y_true, y_true])  # (n_samples, 2)

        # Get full probability matrix
        linear_output = X @ self.weights
        y_pred_full = self.softmax(linear_output)  # (n_samples, 2)

        # Compute gradient
        gradient = X.T @ (y_pred_full - y_true_onehot) / n_samples

        return gradient

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the softmax model with temporary X storage for loss computation.
        """
        # Store X temporarily for loss computation (not ideal but necessary for current design)
        self._temp_X = X_train
        result = super().fit(X_train, y_train, X_val, y_val)
        delattr(self, "_temp_X")
        return result
