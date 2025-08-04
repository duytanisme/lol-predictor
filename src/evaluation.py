import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from models import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelEvaluator:
    """
    Utility class for evaluating and comparing model performance.
    """

    def __init__(self):
        self.results = {}

    def evaluate_model(
        self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a single model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for reporting

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = model.accuracy(y_test, y_pred)

        # Additional metrics
        tp = np.sum((y_test == 1) & (y_pred == 1))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Store results
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        }

        self.results[model_name] = results

        return results

    def print_evaluation_report(self) -> None:
        """
        Print a comprehensive evaluation report for all models.
        """
        print("=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)

        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            print(f"Accuracy:  {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall:    {results['recall']:.4f}")
            print(f"F1-Score:  {results['f1_score']:.4f}")
            print(f"TP: {results['true_positives']}, TN: {results['true_negatives']}")
            print(f"FP: {results['false_positives']}, FN: {results['false_negatives']}")

    def plot_training_history(
        self, models: Dict[str, BaseModel], save_path: Optional[str] = None
    ) -> None:
        """
        Plot training history for all models.

        Args:
            models: Dictionary of model_name -> model
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training loss
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")

        for model_name, model in models.items():
            if hasattr(model, "training_history") and "loss" in model.training_history:
                axes[0].plot(model.training_history["loss"], label=model_name)

        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot training accuracy
        axes[1].set_title("Training Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")

        for model_name, model in models.items():
            if (
                hasattr(model, "training_history")
                and "accuracy" in model.training_history
            ):
                axes[1].plot(model.training_history["accuracy"], label=model_name)

        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_confusion_matrices(
        self,
        models: Dict[str, BaseModel],
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrices for all models.

        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot (optional)
        """
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)

            # Calculate confusion matrix
            tp = np.sum((y_test == 1) & (y_pred == 1))
            tn = np.sum((y_test == 0) & (y_pred == 0))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            fn = np.sum((y_test == 1) & (y_pred == 0))

            confusion_matrix = np.array([[tn, fp], [fn, tp]])

            # Plot heatmap
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Predicted Loss", "Predicted Win"],
                yticklabels=["Actual Loss", "Actual Win"],
                ax=axes[idx],
            )
            axes[idx].set_title(f"{model_name} Confusion Matrix")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def compare_models(self) -> None:
        """
        Create a comparison chart of all models.
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_model first.")
            return

        model_names = list(self.results.keys())
        metrics = ["accuracy", "precision", "recall", "f1_score"]

        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(model_names))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in model_names]
            ax.bar(x + i * width, values, width, label=metric.capitalize())

        ax.set_xlabel("Models")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class HyperparameterTuner:
    """
    Simple hyperparameter tuning utility.
    """

    def __init__(self):
        self.best_params = {}
        self.best_scores = {}

    def grid_search(
        self,
        model_class,
        param_grid: Dict[str, List],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Dict, float]:
        """
        Perform grid search for hyperparameter tuning.

        Args:
            model_class: Model class to instantiate
            param_grid: Dictionary of parameter names and their possible values
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Best parameters and best validation accuracy
        """
        best_params: Optional[Dict] = None
        best_score = -np.inf

        # Generate all parameter combinations
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))

            # Train model with these parameters
            model = model_class(**params)
            model.fit(X_train, y_train, X_val, y_val)

            # Evaluate on validation set
            val_pred = model.predict(X_val)
            val_accuracy = model.accuracy(y_val, val_pred)

            if val_accuracy > best_score:
                best_score = val_accuracy
                best_params = params.copy()

            logger.info(f"Params: {params}, Val Accuracy: {val_accuracy:.4f}")

        if best_params is None:
            raise ValueError("No valid parameter combination found")

        return best_params, best_score

    def tune_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        Tune hyperparameters for all model types.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary of best parameters for each model
        """
        from models import LinearRegression, LogisticRegression, SoftmaxRegression

        # Define parameter grids
        param_grids = {
            "LinearRegression": {
                "learning_rate": [0.001, 0.01, 0.1],
                "batch_size": [16, 32, 64],
                "max_epochs": [500, 1000],
            },
            "LogisticRegression": {
                "learning_rate": [0.001, 0.01, 0.1],
                "batch_size": [16, 32, 64],
                "max_epochs": [500, 1000],
            },
            "SoftmaxRegression": {
                "learning_rate": [0.001, 0.01, 0.1],
                "batch_size": [16, 32, 64],
                "max_epochs": [500, 1000],
            },
        }

        model_classes = {
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "SoftmaxRegression": SoftmaxRegression,
        }

        best_params_all = {}

        for model_name in model_classes.keys():
            print(f"\nTuning {model_name}...")
            best_params, best_score = self.grid_search(
                model_classes[model_name],
                param_grids[model_name],
                X_train,
                y_train,
                X_val,
                y_val,
            )

            best_params_all[model_name] = best_params
            self.best_params[model_name] = best_params
            self.best_scores[model_name] = best_score

            print(f"Best {model_name} params: {best_params}")
            print(f"Best {model_name} score: {best_score:.4f}")

        return best_params_all
