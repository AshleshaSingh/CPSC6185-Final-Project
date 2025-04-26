"""
app.py

Streamlit app for Household Energy Efficiency Classification
===========================================================
This Streamlit app predicts household energy efficiency using a Decision Tree and refines
predictions with a Fuzzy Logic system. It visualizes the Decision Tree structure, fuzzy
membership functions, and sample preprocessed data.

Inputs:
- Household feature values via sidebar inputs
Outputs:
- Decision Tree prediction, Fuzzy Logic score/class, final class
- Visualizations: Decision Tree plot, fuzzy membership functions, sample data

Dependencies: streamlit, pandas, numpy, scikit-learn, scikit-fuzzy, matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import os

# Paths
MODEL_PATH = "models/decision_tree_model.pkl"
DATA_PATH = "data/processed/merged_cleaned.csv"

# Load model
model = joblib.load(MODEL_PATH)
features = model.feature_names_in_

# Load processed data for sample display
data = pd.read_csv(DATA_PATH)

# Streamlit Page Title
st.title("Energy Efficiency Classifier")
st.markdown("Predicts and refines household energy efficiency using Decision Tree and Fuzzy Logic.")

# Sidebar for Input Features
st.sidebar.header("Input Household Characteristics")

# Generate dynamic inputs with realistic ranges and unique keys
input_data = {}
for feature in features:
    if "CLIMATE_" in feature:
        input_data[feature] = st.sidebar.selectbox(f"{feature}", [0, 1], key=f"select_{feature}")
    elif "ENERGY_CONSUMPTION_PER_SQFT" in feature:
        input_data[feature] = st.sidebar.slider(
            f"{feature} (kWh/sqft)", 20.0, 100.0, 50.0, key=f"slider_{feature}"
        )
    else:
        input_data[feature] = st.sidebar.slider(
            f"{feature} (%)", 0.0, 100.0, 50.0, key=f"slider_{feature}"
        )

input_df = pd.DataFrame([input_data])

# --- Decision Tree Prediction ---
tree_pred = model.predict(input_df)[0]

# --- Fuzzy Logic System ---
def fuzzy_system(consumption_value, income_high_value, climate_cold_flag):
    # Define fuzzy variables
    consumption = ctrl.Antecedent(np.arange(20, 100, 1), 'consumption')
    income_high = ctrl.Antecedent(np.arange(0, 100, 1), 'income_high')
    climate_cold = ctrl.Antecedent(np.arange(0, 2, 1), 'climate_cold')
    efficiency = ctrl.Consequent(np.arange(0, 100, 1), 'efficiency')

    # Membership functions
    consumption['low'] = fuzz.trimf(consumption.universe, [20, 20, 50])
    consumption['medium'] = fuzz.trimf(consumption.universe, [30, 55, 80])
    consumption['high'] = fuzz.trimf(consumption.universe, [60, 100, 100])

    income_high['low'] = fuzz.trimf(income_high.universe, [0, 0, 30])
    income_high['medium'] = fuzz.trimf(income_high.universe, [20, 50, 80])
    income_high['high'] = fuzz.trimf(income_high.universe, [60, 100, 100])

    climate_cold['no'] = fuzz.trimf(climate_cold.universe, [0, 0, 1])
    climate_cold['yes'] = fuzz.trimf(climate_cold.universe, [1, 1, 1])

    efficiency['low'] = fuzz.trimf(efficiency.universe, [0, 0, 40])
    efficiency['moderate'] = fuzz.trimf(efficiency.universe, [30, 50, 70])
    efficiency['high'] = fuzz.trimf(efficiency.universe, [60, 100, 100])

    # Rules
    rules = [
        ctrl.Rule(consumption['low'] & climate_cold['yes'] & income_high['high'], efficiency['high']),
        ctrl.Rule(consumption['high'] & income_high['low'], efficiency['low']),
        ctrl.Rule(consumption['medium'] & income_high['medium'], efficiency['moderate']),
        ctrl.Rule(consumption['high'] & climate_cold['no'], efficiency['low']),
        ctrl.Rule(consumption['medium'] & climate_cold['yes'], efficiency['moderate']),
    ]

    # Create control system
    eff_ctrl = ctrl.ControlSystem(rules)
    eff_sim = ctrl.ControlSystemSimulation(eff_ctrl)

    # Pass inputs
    eff_sim.input['consumption'] = consumption_value
    eff_sim.input['income_high'] = income_high_value
    eff_sim.input['climate_cold'] = climate_cold_flag

    eff_sim.compute()

    # Output fuzzy score
    score = eff_sim.output['efficiency']
    if score >= 66:
        fuzzy_class = "High"
    elif score >= 33:
        fuzzy_class = "Moderate"
    else:
        fuzzy_class = "Low"
    return score, fuzzy_class

# Take inputs for fuzzy logic
st.sidebar.header("Input for Fuzzy Scoring")
consumption_input = st.sidebar.slider("ENERGY_CONSUMPTION_PER_SQFT (kWh/sqft)", 20.0, 100.0, 50.0)
income_input = input_data.get("Pct_INCOME_MORE_THAN_150K", 50.0)
climate_cold_flag = input_data.get("CLIMATE_Cold", 0)

# Predict fuzzy score
fuzzy_score, fuzzy_pred = fuzzy_system(consumption_input, income_input, climate_cold_flag)

# --- Final Output Logic ---
final_class = fuzzy_pred if tree_pred != fuzzy_pred else tree_pred

# --- Display Results ---
st.header("Prediction Results")
st.success(f"**Decision Tree Class:** {tree_pred}")
st.info(f"**Fuzzy Logic Class:** {fuzzy_pred} (Score: {fuzzy_score:.2f})")
st.subheader(f"Final Efficiency Classification: {final_class}")

# --- Visualize Decision Tree ---
st.header("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(15, 8))
plot_tree(model, feature_names=features, class_names=["Low", "Moderate", "High"], filled=True)
st.pyplot(fig)

# --- Visualize Fuzzy Membership Functions ---
st.header("Fuzzy Membership Functions")
x_consumption = np.arange(20, 100, 1)
consumption_low = fuzz.trimf(x_consumption, [20, 20, 50])
consumption_medium = fuzz.trimf(x_consumption, [30, 55, 80])
consumption_high = fuzz.trimf(x_consumption, [60, 100, 100])
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_consumption, consumption_low, label='Low')
ax.plot(x_consumption, consumption_medium, label='Medium')
ax.plot(x_consumption, consumption_high, label='High')
ax.set_xlabel('Energy Consumption (kWh/sqft)')
ax.set_ylabel('Membership')
ax.set_title('Consumption Membership Functions')
ax.legend()
st.pyplot(fig)

# --- Sample Data Table ---
st.header("Sample Preprocessed Data")
st.dataframe(data.head())
