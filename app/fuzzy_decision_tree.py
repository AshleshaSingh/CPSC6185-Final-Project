import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.tree import DecisionTreeClassifier

# FuzzyDecisionTree class
class FuzzyDecisionTree:
    def __init__(self, max_depth=6, random_state=42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.feature_names = []
        self.energy_params = None
        self.income_params = None

    def fuzzify_features(self, data):
        energy_fuzz = data['ENERGY_CONSUMPTION_PER_SQFT'].apply(
            lambda x: fuzz_energy(x, *self.energy_params)
        )
        income_fuzz = data['Pct_INCOME_MORE_THAN_150K'].apply(
            lambda x: fuzz_income(x, *self.income_params)
        )
        
        fuzzified_data = pd.DataFrame({
            'energy_low': energy_fuzz.apply(lambda x: x['low']),
            'energy_medium': energy_fuzz.apply(lambda x: x['medium']),
            'energy_high': energy_fuzz.apply(lambda x: x['high']),
            'income_low': income_fuzz.apply(lambda x: x['low']),
            'income_medium': income_fuzz.apply(lambda x: x['medium']),
            'income_high': income_fuzz.apply(lambda x: x['high'])
        })
        
        # Include raw/binary features
        other_features = [
            'CLIMATE_Cold', 'Pct_MAIN_HEAT_AGE_OLDER_THAN_20',
            'CLIMATE_Hot-Humid', 'CLIMATE_Mixed-Humid', 'CLIMATE_Very-Cold',
            'Pct_HOUSING_SINGLE_FAMILY_HOME_DETACHED',
            'Pct_HOUSING_APT_MORE_THAN_5_UNITS', 'Pct_BUILT_BEFORE_1950',
            'Pct_MAIN_AC_AGE_OLDER_THAN_20', 'Pct_MAIN_WATER_HEAT_OLDER_THAN_20'
        ]
        other_features = [f for f in other_features if f in data.columns]
        fuzzified_data = pd.concat([fuzzified_data, data[other_features]], axis=1)
        
        return fuzzified_data

    def fit(self, X, y, energy_params, income_params):
        self.energy_params = energy_params
        self.income_params = income_params
        X_fuzz = self.fuzzify_features(X)
        self.feature_names = [str(f) for f in X_fuzz.columns.tolist()]  # Ensure strings
        self.model.fit(X_fuzz, y)
        return self

    def predict(self, X):
        X_fuzz = self.fuzzify_features(X)
        return self.model.predict(X_fuzz)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def feature_names_in_(self):
        return np.array([str(f) for f in self.feature_names])  # Ensure strings
    
# Fuzzy membership functions
def fuzz_energy(val, min_val, mean_val, max_val):
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
    x = np.linspace(min_val, max_val, 100)
    low = fuzz.trimf(x, [min_val, min_val, mean_val])
    medium = fuzz.trimf(x, [min_val, mean_val, max_val])
    high = fuzz.trimf(x, [mean_val, max_val, max_val])
    return {
        'low': fuzz.interp_membership(x, low, val),
        'medium': fuzz.interp_membership(x, medium, val),
        'high': fuzz.interp_membership(x, high, val)
    }