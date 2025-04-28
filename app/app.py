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
from sklearn.tree import DecisionTreeClassifier
import skfuzzy as fuzz
import os
from fuzzy_decision_tree import FuzzyDecisionTree

# Paths
MODEL_PATH = "models/fuzzy_decision_tree_model.pkl"
DATA_PATH = "data/processed/merged_cleaned.csv"

# Load model and data
try:
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("Model or data files are missing.")
    st.stop()

# Load Efficiency_Class for distribution
EFFICIENCY_PATH = "data/processed/merged_with_efficiency.csv"
try:
    efficiency_data = pd.read_csv(EFFICIENCY_PATH)[['Efficiency_Class']]
except FileNotFoundError:
    efficiency_data = pd.DataFrame({'Efficiency_Class': ['Moderate'] * len(data)})  # Fallback

# Compute dynamic ranges for fuzzy logic
energy_params = (
    data['ENERGY_CONSUMPTION_PER_SQFT'].min(),
    data['ENERGY_CONSUMPTION_PER_SQFT'].mean(),
    data['ENERGY_CONSUMPTION_PER_SQFT'].max()
)
income_params = (
    data['Pct_INCOME_MORE_THAN_150K'].min(),
    data['Pct_INCOME_MORE_THAN_150K'].mean(),
    data['Pct_INCOME_MORE_THAN_150K'].max()
)
equipment_params = (
    data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'].min(),
    data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'].mean(),
    data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'].max()
)

# Define all raw features required by FuzzyDecisionTree
required_raw_features = [
    'ENERGY_CONSUMPTION_PER_SQFT', 'Pct_INCOME_MORE_THAN_150K',
    'CLIMATE_Cold', 'Pct_MAIN_HEAT_AGE_OLDER_THAN_20',
    'CLIMATE_Hot-Humid', 'CLIMATE_Mixed-Humid', 'CLIMATE_Very-Cold',
    'Pct_HOUSING_SINGLE_FAMILY_HOME_DETACHED',
    'Pct_HOUSING_APT_MORE_THAN_5_UNITS', 'Pct_BUILT_BEFORE_1950',
    'Pct_MAIN_AC_AGE_OLDER_THAN_20'
]

# Compute default values (medians) for all features
default_values = {f: data[f].median() if f in data.columns else 0 for f in required_raw_features}

# Verify model features
model_features = model.feature_names_in_
print("Model expected features:", model_features.tolist())  # Debug

# Streamlit Page Title and Introduction
st.title("Home Energy Efficiency Checker")
st.markdown("""
This tool helps you find out how energy-efficient a home might be, based on a few simple details. Adjust the sliders in the sidebar to see if the home is **High**, **Moderate**, or **Low** in energy efficiency. Built using U.S. household data (RECS 2020).

**How to Use:**
1. Set details like energy use, income level, climate, and heating equipment age in the sidebar.
2. View instant results, including a confidence score and reasons for the prediction.
3. Explore charts and data in the "Learn More" section for deeper insights.

*Note*: Enter realistic values. You’ll see a warning if they’re outside typical ranges.
""")

# Sidebar for Input Features
st.sidebar.header("Enter Home Details")
st.sidebar.markdown("Adjust these to describe the home (averages for a state/region).")

# Input fields with validation
input_data = {}
with st.sidebar:
    input_data['ENERGY_CONSUMPTION_PER_SQFT'] = st.slider(
        "Energy Use per Square Foot",
        float(energy_params[0]), float(energy_params[2]), float(energy_params[1]),
        key="slider_energy_use",
        help=f"Annual electricity use per square foot. Range: {energy_params[0]:.1f} to {energy_params[2]:.1f}."
    )
    if input_data['ENERGY_CONSUMPTION_PER_SQFT'] < energy_params[0] or input_data['ENERGY_CONSUMPTION_PER_SQFT'] > energy_params[2]:
        st.warning(f"Energy Use should be between {energy_params[0]:.1f} and {energy_params[2]:.1f}.")
    
    input_data['Pct_INCOME_MORE_THAN_150K'] = st.slider(
        "High-Income Households (%)",
        float(income_params[0]), float(income_params[2]), float(income_params[1]),
        key="slider_income",
        help=f"Percentage of households earning over $150,000. Range: {income_params[0]:.1f}% to {income_params[2]:.1f}%."
    )
    if input_data['Pct_INCOME_MORE_THAN_150K'] < income_params[0] or input_data['Pct_INCOME_MORE_THAN_150K'] > income_params[2]:
        st.warning(f"Income percentage should be between {income_params[0]:.1f}% and {income_params[2]:.1f}%.")
    
    input_data['CLIMATE_Cold'] = st.selectbox(
        "Is the Home in a Cold Climate?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="select_climate",
        help="Select 'Yes' for regions with frequent snow or freezing winters."
    )
    
    input_data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'] = st.slider(
        "Old Heating Equipment (%)",
        float(equipment_params[0]), float(equipment_params[2]), float(equipment_params[1]),
        key="slider_heater_age",
        help=f"Percentage of homes with heating equipment over 20 years old. Range: {equipment_params[0]:.1f}% to {equipment_params[2]:.1f}%."
    )
    if input_data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'] < equipment_params[0] or input_data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'] > equipment_params[2]:
        st.warning(f"Equipment Age percentage should be between {equipment_params[0]:.1f}% and {equipment_params[2]:.1f}%.")

# Prepare input DataFrame for FuzzyDecisionTree
input_df = pd.DataFrame([input_data])
for feature in required_raw_features:
    if feature not in input_df.columns:
        input_df[feature] = default_values.get(feature, 0)  # Use median or 0 for binary features
input_df.columns = input_df.columns.astype(str)  # Ensure string column names
print("input_df columns:", input_df.columns.tolist())  # Debug

# Real-time predictions
fuzzy_inputs = {
    'energy': input_data['ENERGY_CONSUMPTION_PER_SQFT'],
    'income': input_data['Pct_INCOME_MORE_THAN_150K'],
    'climate_cold': input_data['CLIMATE_Cold'],
    'equipment_age': input_data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20']
}

try:
    tree_pred = model.predict(input_df)[0]
except Exception as e:
    st.error(f"Error calculating prediction: {e}")
    st.stop()

# Fuzzy Logic System
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

def fuzzy_system(energy, income, climate_cold, equipment_age):
    fuzz_e = fuzz_energy(energy, *energy_params)
    fuzz_i = fuzz_income(income, *income_params)
    
    score = {'low': 0, 'medium': 0, 'high': 0}
    activated_rules = []
    
    if climate_cold == 1:
        score['high'] += fuzz_e['low'] * 0.5
        if fuzz_e['low'] > 0.3:
            activated_rules.append("Uses low energy in a cold climate, so likely very efficient.")
    
    if climate_cold == 0:
        score['low'] += fuzz_e['high'] * 0.5
        if fuzz_e['high'] > 0.3:
            activated_rules.append("Uses high energy in a non-cold climate, so likely less efficient.")
    
    score['medium'] += fuzz_i['medium'] * 0.3
    if fuzz_i['medium'] > 0.3:
        activated_rules.append("Has a moderate income level, suggesting average efficiency.")
    
    if equipment_age > equipment_params[1]:
        score['low'] += 0.4
        if equipment_age > equipment_params[1]:
            activated_rules.append("Has older heating equipment, which reduces efficiency.")
    
    if fuzz_e['low'] > 0 and fuzz_i['high'] > 0:
        score['high'] += (fuzz_e['low'] * fuzz_i['high']) * 0.5
        if fuzz_e['low'] > 0.3 and fuzz_i['high'] > 0.3:
            activated_rules.append("Uses low energy and has high income, so likely very efficient.")
    
    total = sum(score.values())
    if total == 0:
        score = {'low': 33.33, 'medium': 33.33, 'high': 33.34}
    else:
        for k in score:
            score[k] = score[k] / total * 100
    
    fuzzy_class = max(score, key=score.get).capitalize()
    fuzzy_score = score[fuzzy_class.lower()]
    
    return fuzzy_score, fuzzy_class, activated_rules

# Real-time predictions
try:
    fuzzy_score, fuzzy_pred, activated_rules = fuzzy_system(**fuzzy_inputs)
except Exception as e:
    st.error(f"Error calculating efficiency score: {e}")
    st.stop()

# Final Output
final_class = fuzzy_pred if fuzzy_score > 60 else tree_pred

# Display Results
st.header("Your Home’s Energy Efficiency")
st.markdown("Based on your inputs, here’s the predicted energy efficiency level and confidence score.")

col1, col2 = st.columns(2)
with col1:
    st.success(f"**Efficiency Level**\n\n{final_class}")
with col2:
    st.info(f"**Confidence Score**\n\n{fuzzy_score:.0f}%")

# Why This Prediction?
st.header("Why This Prediction?")
st.markdown("The app uses your inputs to determine efficiency. Here are the key factors:")
if activated_rules:
    for rule in activated_rules:
        st.write(f"- {rule}")
else:
    st.write("- No strong patterns detected. Prediction based on overall trends.")

# Extra Details
st.header("Explore Insights")
with st.expander("How Your Energy Use Compares"):
    st.markdown("This chart shows where your energy use fits compared to typical homes. Lower energy use suggests higher efficiency.")
    x_energy = np.linspace(energy_params[0], energy_params[2], 100)
    energy_low = fuzz.trimf(x_energy, [energy_params[0], energy_params[0], energy_params[1]])
    energy_medium = fuzz.trimf(x_energy, [energy_params[0], energy_params[1], energy_params[2]])
    energy_high = fuzz.trimf(x_energy, [energy_params[1], energy_params[2], energy_params[2]])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_energy, energy_low, label='Low (Efficient)')
    ax.plot(x_energy, energy_medium, label='Medium')
    ax.plot(x_energy, energy_high, label='High (Less Efficient)')
    ax.axvline(x=input_data['ENERGY_CONSUMPTION_PER_SQFT'], color='red', linestyle='--', label='Your Input')
    ax.set_xlabel('Energy Use per Square Foot')
    ax.set_ylabel('Category Fit')
    ax.set_title('Your Energy Use')
    ax.legend()
    st.pyplot(fig)

with st.expander("Efficiency Class Distribution"):
    st.markdown("This shows how efficiency levels (High, Moderate, Low) are distributed across states in the dataset.")
    class_counts = efficiency_data['Efficiency_Class'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    class_counts.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
    ax.set_xlabel('Efficiency Class')
    ax.set_ylabel('Number of States')
    ax.set_title('Distribution of Efficiency Classes')
    st.pyplot(fig)

with st.expander("Feature Comparison (Radar Chart)"):
    st.markdown("This radar chart compares your inputs to the average values in the dataset, highlighting efficiency drivers.")
    features = ['ENERGY_CONSUMPTION_PER_SQFT', 'Pct_INCOME_MORE_THAN_150K', 'Pct_MAIN_HEAT_AGE_OLDER_THAN_20']
    labels = ['Energy Use', 'High Income', 'Old Equipment']
    user_values = [input_data[f] for f in features]
    avg_values = [data[f].mean() for f in features]
    
    # Normalize values to 0-1 for radar chart
    max_values = [max(data[f].max(), input_data[f]) for f in features]
    user_norm = [v / m for v, m in zip(user_values, max_values)]
    avg_norm = [v / m for v, m in zip(avg_values, max_values)]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    user_norm += user_norm[:1]
    avg_norm += avg_norm[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, user_norm, color='blue', alpha=0.25, label='Your Input')
    ax.plot(angles, user_norm, color='blue', linewidth=2)
    ax.fill(angles, avg_norm, color='green', alpha=0.25, label='Dataset Average')
    ax.plot(angles, avg_norm, color='green', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Your Inputs vs. Dataset Averages')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    st.pyplot(fig)

with st.expander("Decision Tree Structure"):
    st.markdown("This shows the decision-making process of the model, illustrating how inputs lead to efficiency predictions.")
    fig, ax = plt.subplots(figsize=(12, 8))
    if hasattr(model, 'model') and isinstance(model.model, DecisionTreeClassifier):
        # FuzzyDecisionTree case
        plot_tree(model.model, feature_names=model.feature_names_in_, class_names=['High', 'Moderate', 'Low'], filled=True, ax=ax)
    else:
        # Standard DecisionTreeClassifier case
        plot_tree(model, feature_names=model.feature_names_in_, class_names=['High', 'Moderate', 'Low'], filled=True, ax=ax)
    st.pyplot(fig)

with st.expander("Typical Home Data"):
    st.markdown("Compare your inputs to typical homes in the dataset.")
    st.dataframe(data[required_raw_features].head().rename(columns={
        'ENERGY_CONSUMPTION_PER_SQFT': 'Energy Use per Sqft',
        'Pct_INCOME_MORE_THAN_150K': 'High-Income Households (%)',
        'CLIMATE_Cold': 'Cold Climate (0=No, 1=Yes)',
        'Pct_MAIN_HEAT_AGE_OLDER_THAN_20': 'Old Heating Equipment (%)',
        'CLIMATE_Hot-Humid': 'Hot-Humid Climate',
        'CLIMATE_Mixed-Humid': 'Mixed-Humid Climate',
        'CLIMATE_Very-Cold': 'Very Cold Climate',
        'Pct_HOUSING_SINGLE_FAMILY_HOME_DETACHED': 'Single-Family Homes (%)',
        'Pct_HOUSING_APT_MORE_THAN_5_UNITS': 'Large Apartments (%)',
        'Pct_BUILT_BEFORE_1950': 'Homes Built Before 1950 (%)',
        'Pct_MAIN_AC_AGE_OLDER_THAN_20': 'Old Air Conditioning (%)'
    }), use_container_width=True)

with st.expander("What Matters Most"):
    st.markdown("These factors most influence the efficiency prediction, based on the model’s analysis.")
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    })
    feature_importance['Feature'] = feature_importance['Feature'].map({
        'energy_low': 'Low Energy Use',
        'energy_medium': 'Medium Energy Use',
        'energy_high': 'High Energy Use',
        'income_low': 'Low Income Households',
        'income_medium': 'Medium Income Households',
        'income_high': 'High Income Households',
        'CLIMATE_Cold': 'Cold Climate',
        'Pct_MAIN_HEAT_AGE_OLDER_THAN_20': 'Old Heating Equipment',
        'CLIMATE_Hot-Humid': 'Hot-Humid Climate',
        'CLIMATE_Mixed-Humid': 'Mixed-Humid Climate',
        'CLIMATE_Very-Cold': 'Very Cold Climate',
        'Pct_HOUSING_SINGLE_FAMILY_HOME_DETACHED': 'Single-Family Homes (%)',
        'Pct_HOUSING_APT_MORE_THAN_5_UNITS': 'Large Apartments (%)',
        'Pct_BUILT_BEFORE_1950': 'Homes Built Before 1950 (%)',
        'Pct_MAIN_AC_AGE_OLDER_THAN_20': 'Old Air Conditioning (%)'
    }).fillna(feature_importance['Feature'])
    feature_importance['Importance'] = feature_importance['Importance'].round(3)
    feature_importance = feature_importance[feature_importance['Importance'] > 0]
    st.dataframe(feature_importance, use_container_width=True)

