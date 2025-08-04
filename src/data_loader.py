import logging
import numpy as np
import os
import pandas as pd

from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataLoader:
    """
    Data loader for League of Legends game state data.
    Handles loading, preprocessing, and splitting of the dataset.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the dataset file
        """
        self.data_path = data_path
        self.features = [
            "side",
            "first_blood",
            "first_dragon",
            "first_herald",
            "first_baron",
            "first_tower",
            "first_mid_tower",
            "first_to_3_towers",
            "turret_plates",
            "opp_turret_plates",
            "gold_diff_at_10",
            "xp_diff_at_10",
            "cs_diff_at_10",
            "kills_at_10",
            "assists_at_10",
            "opp_kills_at_10",
            "opp_assists_at_10",
            "gold_diff_at_15",
            "xp_diff_at_15",
            "cs_diff_at_15",
            "kills_at_15",
            "assists_at_15",
            "opp_kills_at_15",
            "opp_assists_at_15",
        ]
        self.target = "result"  # 1 for win, 0 for loss

    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic League of Legends game data for development.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic game data
        """
        np.random.seed(42)

        data = {}

        # Binary features (0 or 1)
        binary_features = [
            "side",
            "first_blood",
            "first_dragon",
            "first_herald",
            "first_baron",
            "first_tower",
            "first_mid_tower",
            "first_to_3_towers",
        ]

        for feature in binary_features:
            data[feature] = np.random.binomial(1, 0.5, n_samples)

        # Turret plates (0-5 typically)
        data["turret_plates"] = np.random.poisson(2, n_samples)
        data["opp_turret_plates"] = np.random.poisson(2, n_samples)

        # Gold/XP/CS differences (can be negative)
        data["gold_diff_at_10"] = np.random.normal(0, 1000, n_samples)
        data["xp_diff_at_10"] = np.random.normal(0, 500, n_samples)
        data["cs_diff_at_10"] = np.random.normal(0, 10, n_samples)

        data["gold_diff_at_15"] = np.random.normal(0, 1500, n_samples)
        data["xp_diff_at_15"] = np.random.normal(0, 750, n_samples)
        data["cs_diff_at_15"] = np.random.normal(0, 15, n_samples)

        # Kills and assists (non-negative)
        data["kills_at_10"] = np.random.poisson(2, n_samples)
        data["assists_at_10"] = np.random.poisson(3, n_samples)
        data["opp_kills_at_10"] = np.random.poisson(2, n_samples)
        data["opp_assists_at_10"] = np.random.poisson(3, n_samples)

        data["kills_at_15"] = np.random.poisson(4, n_samples)
        data["assists_at_15"] = np.random.poisson(6, n_samples)
        data["opp_kills_at_15"] = np.random.poisson(4, n_samples)
        data["opp_assists_at_15"] = np.random.poisson(6, n_samples)

        # Create target variable with some correlation to features
        # Teams with positive gold/xp differences and more objectives are more likely to win
        win_probability = (
            0.5
            + 0.1 * data["first_blood"]
            + 0.1 * data["first_dragon"]
            + 0.1 * data["first_tower"]
            + 0.0005 * np.clip(data["gold_diff_at_15"], -2000, 2000)
            + 0.001 * np.clip(data["xp_diff_at_15"], -1000, 1000)
        )
        win_probability = np.clip(win_probability, 0.1, 0.9)

        data["result"] = np.random.binomial(1, win_probability, n_samples)

        return pd.DataFrame(data)

    def load_data(self) -> pd.DataFrame:
        """
        Load data from file or generate synthetic data.

        Returns:
            DataFrame with game data
        """
        if self.data_path and os.path.exists(self.data_path):
            return pd.read_csv(self.data_path)
        else:
            logging.info("No data file found, generating synthetic data")
            return self.generate_synthetic_data()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data (normalization, feature engineering, etc.).

        Args:
            df: Raw dataframe

        Returns:
            Preprocessed dataframe
        """
        df_processed = df.copy()

        # Ensure all features are present
        for feature in self.features:
            if feature not in df_processed.columns:
                logging.warning(f"Feature {feature} not found in data, setting to 0")
                df_processed[feature] = 0

        # Normalize continuous features
        continuous_features = [
            "gold_diff_at_10",
            "xp_diff_at_10",
            "cs_diff_at_10",
            "gold_diff_at_15",
            "xp_diff_at_15",
            "cs_diff_at_15",
        ]

        for feature in continuous_features:
            if feature in df_processed.columns:
                mean = df_processed[feature].mean()
                std = df_processed[feature].std()
                if std != 0:
                    df_processed[feature] = (df_processed[feature] - mean) / std

        return df_processed

    def split_data(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Preprocessed dataframe
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Shuffle the data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        n_samples = len(df_shuffled)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        # Split indices
        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, None)

        # Extract features and target
        X = df_shuffled[self.features].values.astype(np.float64)
        y = df_shuffled[self.target].values.astype(np.float64)

        # Add bias term (column of ones) to features
        X = np.column_stack([np.ones(X.shape[0]), X])

        return (
            X[train_idx],
            X[val_idx],
            X[test_idx],
            y[train_idx],
            y[val_idx],
            y[test_idx],
        )
