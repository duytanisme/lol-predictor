"""
League of Legends Game Outcome Predictor

This script implements and compares three regression models for predicting
League of Legends game outcomes based on early game state features.

Models implemented:
1. Linear Regression with MSE loss
2. Logistic Regression with Cross-entropy loss
3. Softmax Regression with Cross-entropy loss

The script loads/generates data, trains models, evaluates performance,
and provides visualizations for model comparison.
"""

import logging
import numpy as np
from typing import Dict, Any

from data_loader import DataLoader
from evaluation import ModelEvaluator, HyperparameterTuner
from models import LinearRegression, LogisticRegression, SoftmaxRegression, BaseModel
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    """
    Main execution function for the LoL predictor project.
    """
    logger.info("Starting League of Legends Game Outcome Predictor")

    # =====================================
    # 1. DATA LOADING AND PREPROCESSING
    # =====================================
    logger.info("Loading and preprocessing data...")

    # Initialize data loader
    data_loader = DataLoader()

    # Load or generate data
    df = data_loader.load_data()
    logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")

    # Preprocess data
    df_processed = data_loader.preprocess_data(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
        df_processed
    )

    logger.info(
        f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    logger.info(f"Feature dimensionality: {X_train.shape[1]} (including bias term)")

    # =====================================
    # 2. MODEL TRAINING
    # =====================================
    logger.info("Training models...")

    # Initialize models with default hyperparameters
    models = {
        "Linear Regression": LinearRegression(
            learning_rate=0.01, batch_size=32, max_epochs=1000
        ),
        "Logistic Regression": LogisticRegression(
            learning_rate=0.01, batch_size=32, max_epochs=1000
        ),
        "Softmax Regression": SoftmaxRegression(
            learning_rate=0.01, batch_size=32, max_epochs=1000
        ),
    }

    # Train each model
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train, X_val, y_val)
        logger.info(f"{model_name} training completed")

    # =====================================
    # 3. MODEL EVALUATION
    # =====================================
    logger.info("Evaluating models...")

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Evaluate each model
    for model_name, model in models.items():
        results = evaluator.evaluate_model(model, X_test, y_test, model_name)
        logger.info(
            f"{model_name} - Test Accuracy: {results['accuracy']:.4f}, "
            f"F1-Score: {results['f1_score']:.4f}"
        )

    # Print comprehensive evaluation report
    evaluator.print_evaluation_report()

    # =====================================
    # 4. VISUALIZATIONS
    # =====================================
    logger.info("Generating visualizations...")

    # try:
    #     # Plot training history
    #     evaluator.plot_training_history(models)

    #     # Plot confusion matrices
    #     evaluator.plot_confusion_matrices(models, X_test, y_test)

    #     # Model comparison chart
    #     evaluator.compare_models()

    # except Exception as e:
    #     logger.warning(f"Visualization error (likely running headless): {e}")
    #     logger.info("Skipping visualizations - data analysis completed successfully")

    # =====================================
    # 5. HYPERPARAMETER TUNING (OPTIONAL)
    # =====================================
    logger.info("Starting hyperparameter tuning...")

    # Initialize tuner
    tuner = HyperparameterTuner()

    try:
        # Perform grid search (this may take a while)
        best_params = tuner.tune_all_models(X_train, y_train, X_val, y_val)

        logger.info("Hyperparameter tuning completed")
        logger.info("Best parameters found:")
        for model_name, params in best_params.items():
            logger.info(f"{model_name}: {params}")

        # Train models with best parameters
        logger.info("Training models with optimized hyperparameters...")

        optimized_models = {}
        for model_name, params in best_params.items():
            model = None
            if model_name == "LinearRegression":
                model = LinearRegression(**params)
            elif model_name == "LogisticRegression":
                model = LogisticRegression(**params)
            elif model_name == "SoftmaxRegression":
                model = SoftmaxRegression(**params)

            if model is not None:
                model.fit(X_train, y_train, X_val, y_val)
                optimized_models[f"{model_name} (Optimized)"] = model

        # Evaluate optimized models
        logger.info("Evaluating optimized models...")
        for model_name, model in optimized_models.items():
            results = evaluator.evaluate_model(model, X_test, y_test, model_name)
            logger.info(
                f"{model_name} - Test Accuracy: {results['accuracy']:.4f}, "
                f"F1-Score: {results['f1_score']:.4f}"
            )

    except Exception as e:
        logger.warning(f"Hyperparameter tuning failed: {e}")
        logger.info("Continuing with default hyperparameters")

    # =====================================
    # 6. SUMMARY AND INSIGHTS
    # =====================================
    print("=" * 60)
    logger.info("ANALYSIS SUMMARY")
    print("=" * 60)

    # Find best performing model
    best_model = None
    best_accuracy = 0
    best_name = ""

    for model_name in evaluator.results:
        accuracy = evaluator.results[model_name]["accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = model_name
            best_model = models.get(model_name.replace(" (Optimized)", ""))

    logger.info(f"Best performing model: {best_name}")
    logger.info(f"Best test accuracy: {best_accuracy:.4f}")

    # Feature importance analysis (for linear models)
    if best_model and hasattr(best_model, "weights"):
        logger.info("Feature importance analysis:")
        feature_names = ["bias"] + data_loader.features
        weights = (
            best_model.weights
            if hasattr(best_model.weights, "__len__")
            else best_model.weights.flatten()
        )

        # Get top 5 most important features (by absolute weight)
        if len(weights) == len(feature_names):
            importance_indices = np.argsort(np.abs(weights))[-6:]  # Top 5 + bias
            logger.info("Top 5 most important features:")
            for i in importance_indices[::-1]:
                if i > 0:  # Skip bias term
                    logger.info(f"  {feature_names[i]}: {weights[i]:.4f}")

    logger.info("Analysis completed successfully!")
    print("=" * 60)


def run_quick_demo():
    """
    Quick demonstration with minimal data for testing purposes.
    """
    logger.info("Running quick demo with small dataset...")

    # Generate small dataset
    data_loader = DataLoader()
    df = data_loader.generate_synthetic_data(n_samples=1000)
    df_processed = data_loader.preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
        df_processed
    )

    # Train single model
    model = LogisticRegression(learning_rate=0.01, batch_size=16, max_epochs=100)
    model.fit(X_train, y_train, X_val, y_val)

    # Quick evaluation
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X_test, y_test, "Logistic Regression")

    logger.info(f"Demo completed - Test Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    # Uncomment the line below for a quick demo instead of full analysis
    # run_quick_demo()

    # Run full analysis
    main()
