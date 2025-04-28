"""
fuzzy_decision_tree.py

This module provides the FuzzyDecisionTree class, which integrates fuzzy logic with a decision tree classifier
for energy efficiency prediction. It fuzzifies continuous features (energy consumption and income) into low,
medium, and high categories using triangular membership functions, then trains a DecisionTreeClassifier on
the fuzzified and raw features. This implementation supports the energy efficiency prediction project,
aligning with the project proposal’s requirement for combining fuzzy logic and decision trees (Section 5.3.3).

Dependencies:
- pandas: Data manipulation and DataFrame operations.
- numpy: Numerical computations for fuzzy logic.
- skfuzzy: Fuzzy logic membership functions.
- sklearn.tree: DecisionTreeClassifier for model training and prediction.
"""

import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.tree import DecisionTreeClassifier


class FuzzyDecisionTree:
    """
    A classifier that combines fuzzy logic with a decision tree for energy efficiency prediction.

    This class fuzzifies continuous features (ENERGY_CONSUMPTION_PER_SQFT, Pct_INCOME_MORE_THAN_150K)
    into fuzzy membership values (low, medium, high) and combines them with raw/binary features to
    train a DecisionTreeClassifier. It is used to predict energy efficiency classes (High, Moderate, Low).

    Attributes:
        model (DecisionTreeClassifier): The underlying decision tree classifier.
        feature_names (list): Names of fuzzified and raw features used in training.
        energy_params (tuple): Min, mean, max values for energy consumption fuzzification.
        income_params (tuple): Min, mean, max values for income fuzzification.
    """

    def __init__(self, max_depth=6, random_state=42):
        """
        Initialize the FuzzyDecisionTree with a DecisionTreeClassifier.

        Args:
            max_depth (int, optional): Maximum depth of the decision tree. Defaults to 6.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.feature_names = []
        self.energy_params = None
        self.income_params = None

    def fuzzify_features(self, data):
        """
        Fuzzify continuous features and combine with raw/binary features.

        Converts ENERGY_CONSUMPTION_PER_SQFT and Pct_INCOME_MORE_THAN_150K into fuzzy membership values
        (low, medium, high) using triangular membership functions. Concatenates these with other
        raw/binary features to create the input for the decision tree.

        Args:
            data (pd.DataFrame): Input data containing ENERGY_CONSUMPTION_PER_SQFT,
                                Pct_INCOME_MORE_THAN_150K, and other features.

        Returns:
            pd.DataFrame: DataFrame with fuzzified features (energy_low, energy_medium, energy_high,
                          income_low, income_medium, income_high) and raw/binary features.
        """
        # Apply fuzzy membership functions to energy consumption
        energy_fuzz = data['ENERGY_CONSUMPTION_PER_SQFT'].apply(
            lambda x: fuzz_energy(x, *self.energy_params)
        )
        # Apply fuzzy membership functions to income percentage
        income_fuzz = data['Pct_INCOME_MORE_THAN_150K'].apply(
            lambda x: fuzz_income(x, *self.income_params)
        )
        
        # Create DataFrame with fuzzified features
        fuzzified_data = pd.DataFrame({
            'energy_low': energy_fuzz.apply(lambda x: x['low']),
            'energy_medium': energy_fuzz.apply(lambda x: x['medium']),
            'energy_high': energy_fuzz.apply(lambda x: x['high']),
            'income_low': income_fuzz.apply(lambda x: x['low']),
            'income_medium': income_fuzz.apply(lambda x: x['medium']),
            'income_high': income_fuzz.apply(lambda x: x['high'])
        })
        
        # List of raw/binary features to include
        other_features = [
            'CLIMATE_Cold', 'Pct_MAIN_HEAT_AGE_OLDER_THAN_20',
            'CLIMATE_Hot-Humid', 'CLIMATE_Mixed-Humid', 'CLIMATE_Very-Cold',
            'Pct_HOUSING_SINGLE_FAMILY_HOME_DETACHED',
            'Pct_HOUSING_APT_MORE_THAN_5_UNITS', 'Pct_BUILT_BEFORE_1950',
            'Pct_MAIN_AC_AGE_OLDER_THAN_20', 'Pct_MAIN_WATER_HEAT_OLDER_THAN_20'
        ]
        # Filter for features present in the input data
        other_features = [f for f in other_features if f in data.columns]
        # Combine fuzzified and raw features
        fuzzified_data = pd.concat([fuzzified_data, data[other_features]], axis=1)
        
        return fuzzified_data

    def fit(self, X, y, energy_params, income_params):
        """
        Train the decision tree on fuzzified features.

        Fuzzifies the input features, stores feature names, and fits the DecisionTreeClassifier.

        Args:
            X (pd.DataFrame): Input features including ENERGY_CONSUMPTION_PER_SQFT,
                              Pct_INCOME_MORE_THAN_150K, and other features.
            y (pd.Series): Target variable (e.g., Efficiency_Class).
            energy_params (tuple): (min, mean, max) values for energy consumption fuzzification.
            income_params (tuple): (min, mean, max) values for income fuzzification.

        Returns:
            self: The fitted FuzzyDecisionTree instance.
        """
        self.energy_params = energy_params
        self.income_params = income_params
        X_fuzz = self.fuzzify_features(X)
        # Store feature names as strings for compatibility
        self.feature_names = [str(f) for f in X_fuzz.columns.tolist()]
        self.model.fit(X_fuzz, y)
        return self

    def predict(self, X):
        """
        Predict class labels using the trained decision tree.

        Fuzzifies the input features and uses the DecisionTreeClassifier to predict class labels.

        Args:
            X (pd.DataFrame): Input features including ENERGY_CONSUMPTION_PER_SQFT,
                              Pct_INCOME_MORE_THAN_150K, and other features.

        Returns:
            np.ndarray: Predicted class labels (e.g., ['High', 'Moderate', 'Low']).
        """
        X_fuzz = self.fuzzify_features(X)
        return self.model.predict(X_fuzz)

    @property
    def feature_importances_(self):
        """
        Get the feature importances from the trained decision tree.

        Returns:
            np.ndarray: Importance scores for each feature.
        """
        return self.model.feature_importances_

    @property
    def feature_names_in_(self):
        """
        Get the names of features used in training.

        Returns:
            np.ndarray: Array of feature names as strings.
        """
        return np.array([str(f) for f in self.feature_names])


def fuzz_energy(val, min_val, mean_val, max_val):
    """
    Compute fuzzy membership values for energy consumption.

    Uses triangular membership functions to categorize an energy consumption value into
    low, medium, and high membership degrees.

    Args:
        val (float): Energy consumption value (ENERGY_CONSUMPTION_PER_SQFT).
        min_val (float): Minimum energy consumption in the dataset.
        mean_val (float): Mean energy consumption in the dataset.
        max_val (float): Maximum energy consumption in the dataset.

    Returns:
        dict: Membership degrees for 'low', 'medium', and 'high' categories.
    """
    x = np.linspace(min_val, max_val, 100)
    low = fuzz.trimf(x, [min_val, min_val, mean_val])
    medium = fuzz.trimf(x, [min_val, mean_val, max_val])
    high = fuzz.trimf(x, [mean_val, max_val, max_val])
    return {
        'low': fuzz.interp_membership(x, low, val),
        'medium': fuzz.interp_membership(x, medium, val),
        'high': fuzz.interp_membership(x, high, val)
    }


def fuzz_income(val, min_val, mean_val, max_val):
    """
    Compute fuzzy membership values for income percentage.

    Uses triangular membership functions to categorize an income percentage value into
    low, medium, and high membership degrees.

    Args:
        val (float): Income percentage value (Pct_INCOME_MORE_THAN_150K).
        min_val (float): Minimum income percentage in the dataset.
        mean_val (float): Mean income percentage in the dataset.
        max_val (float): Maximum income percentage in the dataset.

    Returns:
        dict: Membership degrees for 'low', 'medium', and 'high' categories.
    """
    x = np.linspace(min_val, max_val, 100)
    low = fuzz.trimf(x, [min_val, min_val, mean_val])
    medium = fuzz.trimf(x, [min_val, mean_val, max_val])
    high = fuzz.trimf(x, [mean_val, max_val, max_val])
    return {
        'low': fuzz.interp_membership(x, low, val),
        'medium': fuzz.interp_membership(x, medium, val),
        'high': fuzz.interp_membership(x, high, val)
    }
