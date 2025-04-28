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
import os

# Paths
MODEL_PATH = "models/decision_tree_model.pkl"
DATA_PATH = "data/processed/merged_cleaned.csv"

# Load model and data
try:
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("Model or data files are missing. Please contact support.")
    st.stop()

# Define key features
key_features = [
    'ENERGY_CONSUMPTION_PER_SQFT', 'Pct_INCOME_MORE_THAN_150K',
    'CLIMATE_Cold', 'Pct_MAIN_HEAT_AGE_OLDER_THAN_20'
]

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

# Verify model features
model_features = model.feature_names_in_
missing_features = [f for f in model_features if f not in key_features and f not in [
    'energy_low', 'energy_medium', 'energy_high', 'income_low', 'income_medium', 'income_high'
]]
default_values = {f: data[f].median() for f in missing_features if f in data.columns}

# Streamlit Page Title and Introduction
st.title("Home Energy Efficiency Checker")
st.markdown("""
This tool helps you find out how energy-efficient a home might be, based on a few simple details. Just adjust the sliders in the sidebar, and the app will instantly show you if the home is **High**, **Moderate**, or **Low** in energy efficiency. It’s based on data from U.S. households (RECS 2020).

**How to Use:**
1. Use the sidebar to set details like energy use, income level, climate, and heating equipment age.
2. See the results right away, including a score and reasons for the prediction.
3. Check out extra details (like charts or sample data) if you’re curious, by clicking the expandable sections.

*Note*: Make sure the values you enter are realistic. You’ll get a warning if they’re too far off.
""")

# Sidebar for Input Features
st.sidebar.header("Enter Home Details")
st.sidebar.markdown("Adjust these settings to describe the home. They represent averages for a state or region.")

# Input fields with simplified labels and validation
input_data = {}
with st.sidebar:
    # Energy Consumption
    input_data['ENERGY_CONSUMPTION_PER_SQFT'] = st.slider(
        "Energy Use per Square Foot",
        float(energy_params[0]), float(energy_params[2]), float(energy_params[1]),
        key="slider_energy_use",
        help=f"How much electricity the home uses per square foot each year. Typical range: {energy_params[0]:.1f} to {energy_params[2]:.1f}."
    )
    if input_data['ENERGY_CONSUMPTION_PER_SQFT'] < energy_params[0] or input_data['ENERGY_CONSUMPTION_PER_SQFT'] > energy_params[2]:
        st.warning(f"Please choose an Energy Use value between {energy_params[0]:.1f} and {energy_params[2]:.1f}.")
    
    # Income
    input_data['Pct_INCOME_MORE_THAN_150K'] = st.slider(
        "High-Income Households (%)",
        float(income_params[0]), float(income_params[2]), float(income_params[1]),
        key="slider_income",
        help=f"Percentage of households earning over $150,000 per year. Typical range: {income_params[0]:.1f}% to {income_params[2]:.1f}%."
    )
    if input_data['Pct_INCOME_MORE_THAN_150K'] < income_params[0] or input_data['Pct_INCOME_MORE_THAN_150K'] > income_params[2]:
        st.warning(f"Please choose an Income percentage between {income_params[0]:.1f}% and {income_params[2]:.1f}%.")
    
    # Climate
    input_data['CLIMATE_Cold'] = st.selectbox(
        "Is the Home in a Cold Climate?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="select_climate",
        help="Choose 'Yes' if the home is in a cold region (e.g., frequent snow or freezing winters)."
    )
    
    # Equipment Age
    input_data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'] = st.slider(
        "Old Heating Equipment (%)",
        float(equipment_params[0]), float(equipment_params[2]), float(equipment_params[1]),
        key="slider_heater_age",
        help=f"Percentage of homes with heating equipment over 20 years old. Typical range: {equipment_params[0]:.1f}% to {equipment_params[2]:.1f}%."
    )
    if input_data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'] < equipment_params[0] or input_data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20'] > equipment_params[2]:
        st.warning(f"Please choose an Equipment Age percentage between {equipment_params[0]:.1f}% and {equipment_params[2]:.1f}%.")

# Prepare input DataFrame for FuzzyDecisionTree
input_df = pd.DataFrame([input_data])
for feature in missing_features:
    if feature in default_values:
        input_df[feature] = default_values[feature]
    else:
        input_df[feature] = 0  # Default for binary features

# Fuzzy Logic System (aligned with fuzzy_logic.ipynb)
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
    # Fuzzify inputs
    fuzz_e = fuzz_energy(energy, *energy_params)
    fuzz_i = fuzz_income(income, *income_params)
    
    # Initialize scores
    score = {'low': 0, 'medium': 0, 'high': 0}
    activated_rules = []
    
    # Rule 1: Low energy, Cold climate → High efficiency
    if climate_cold == 1:
        score['high'] += fuzz_e['low'] * 0.5
        if fuzz_e['low'] > 0.3:
            activated_rules.append("Uses low energy in a cold climate, so likely very efficient.")
    
    # Rule 2: High energy, no Cold climate → Low efficiency
    if climate_cold == 0:
        score['low'] += fuzz_e['high'] * 0.5
        if fuzz_e['high'] > 0.3:
            activated_rules.append("Uses high energy in a non-cold climate, so likely less efficient.")
    
    # Rule 3: Medium income → Moderate efficiency
    score['medium'] += fuzz_i['medium'] * 0.3
    if fuzz_i['medium'] > 0.3:
        activated_rules.append("Has a moderate income level, suggesting average efficiency.")
    
    # Rule 4: Old equipment → Low efficiency
    if equipment_age > equipment_params[1]:
        score['low'] += 0.4
        activated_rules.append("Has older heating equipment, which reduces efficiency.")
    
    # Rule 5: Low energy and high income → High efficiency
    if fuzz_e['low'] > 0 and fuzz_i['high'] > 0:
        score['high'] += (fuzz_e['low'] * fuzz_i['high']) * 0.5
        if fuzz_e['low'] > 0.3 and fuzz_i['high'] > 0.3:
            activated_rules.append("Uses low energy and has high income, so likely very efficient.")
    
    # Normalize scores, handle edge case
    total = sum(score.values())
    if total == 0:
        score = {'low': 33.33, 'medium': 33.33, 'high': 33.34}  # Default equal distribution
    else:
        for k in score:
            score[k] = score[k] / total * 100
    
    # Determine fuzzy class
    fuzzy_class = max(score, key=score.get).capitalize()
    fuzzy_score = score[fuzzy_class.lower()]
    
    return fuzzy_score, fuzzy_class, activated_rules

# Map input_data to fuzzy_system parameters
fuzzy_inputs = {
    'energy': input_data['ENERGY_CONSUMPTION_PER_SQFT'],
    'income': input_data['Pct_INCOME_MORE_THAN_150K'],
    'climate_cold': input_data['CLIMATE_Cold'],
    'equipment_age': input_data['Pct_MAIN_HEAT_AGE_OLDER_THAN_20']
}

# Real-time predictions
try:
    tree_pred = model.predict(input_df)[0]
except Exception as e:
    st.error(f"Error calculating prediction: {e}")
    st.stop()

try:
    fuzzy_score, fuzzy_pred, activated_rules = fuzzy_system(**fuzzy_inputs)
except Exception as e:
    st.error(f"Error calculating efficiency score: {e}")
    st.stop()

# Final Output Logic
final_class = fuzzy_pred if fuzzy_score > 60 else tree_pred

# Display Results
st.header("Your Home’s Energy Efficiency")
st.markdown("Based on the details you provided, here’s how energy-efficient the home is likely to be. The Confidence Score shows how strongly the app believes this efficiency level is correct.")

col1, col2 = st.columns(2)
with col1:
    st.success(f"**Efficiency Level**\n\n{final_class}")
with col2:
    st.info(f"**Confidence Score**\n\n{fuzzy_score:.0f}%")

# Why This Prediction?
st.header("Why This Prediction?")
st.markdown("The app looks at your inputs to decide the efficiency level. Here’s what influenced the result:")
if activated_rules:
    for rule in activated_rules:
        st.write(f"- {rule}")
else:
    st.write("- No specific patterns stood out. The prediction is based on overall trends.")

# Extra Details
st.header("Learn More (Optional)")
with st.expander("How Your Energy Use Compares"):
    st.markdown("This chart shows where your energy use fits compared to typical homes. Lower energy use means higher efficiency.")
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

with st.expander("Typical Home Data"):
    st.markdown("Here’s what typical homes in the dataset look like, for comparison.")
    st.dataframe(data[key_features].head().rename(columns={
        'ENERGY_CONSUMPTION_PER_SQFT': 'Energy Use per Sqft',
        'Pct_INCOME_MORE_THAN_150K': 'High-Income Households (%)',
        'CLIMATE_Cold': 'Cold Climate (0=No, 1=Yes)',
        'Pct_MAIN_HEAT_AGE_OLDER_THAN_20': 'Old Heating Equipment (%)'
    }), use_container_width=True)

with st.expander("What Matters Most"):
    st.markdown("""
    Some factors influence the prediction more than others. The table below shows which features the model considers important. Your inputs (like Energy Use) are categorized as Low, Medium, or High, while other features (like building age) are set to typical values from the dataset.
    """)
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
        'Pct_MAIN_AC_AGE_OLDER_THAN_20': 'Old Air Conditioning (%)',
        'Pct_MAIN_WATER_HEAT_OLDER_THAN_20': 'Old Water Heater (%)'
    }).fillna(feature_importance['Feature'])
    feature_importance['Importance'] = feature_importance['Importance'].round(3)
    feature_importance = feature_importance[feature_importance['Importance'] > 0]
    st.dataframe(feature_importance, use_container_width=True)