import fuzzy_logic_utils as FuzzyLogic
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

class FuzzyDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, **tree_params):
        self.tree = DecisionTreeClassifier(**tree_params)
        self.features = ['energy_low', 'energy_medium', 'energy_high',
                         'income_low', 'income_medium', 'income_high',
                         'climate_cold', 'climate_hot_humid', 'climate_mixed_humid']

    def fuzzify_row(self, row):
        energy_fuzzy = FuzzyLogic.fuzz_energy(row['ENERGY_CONSUMPTION_PER_SQFT'])
        income_fuzzy = FuzzyLogic.fuzz_income(row['Pct_INCOME_MORE_THAN_150K'])

        return {
            'energy_low': energy_fuzzy['low'],
            'energy_medium': energy_fuzzy['medium'],
            'energy_high': energy_fuzzy['high'],
            'income_low': income_fuzzy['low'],
            'income_medium': income_fuzzy['medium'],
            'income_high': income_fuzzy['high'],
            'climate_cold': row.get('CLIMATE_Cold', 0),
            'climate_hot_humid': row.get('CLIMATE_Hot-Humid', 0),
            'climate_mixed_humid': row.get('CLIMATE_Mixed-Humid', 0)
        }

    def fuzzify(self, X):
        fuzzy_df = X.apply(self.fuzzify_row, axis=1, result_type='expand')
        return fuzzy_df[self.features]

    def fit(self, X, y):
        X_fuzzy = self.fuzzify(X)
        self.tree.fit(X_fuzzy, y)
        return self

    def predict(self, X):
        X_fuzzy = self.fuzzify(X)
        return self.tree.predict(X_fuzzy)

    def rules(self):
        return export_text(self.tree, feature_names=self.features)
