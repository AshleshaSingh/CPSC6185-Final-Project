import streamlit as st
import pandas as pd
from fuzzy_decision_tree import FuzzyDecisionTree

# Load your cleaned dataset
df = pd.read_csv("data/processed/merged_with_efficiency.csv")

# Helper: Categorize EFFICIENCY_SCORE into 0, 1, 2 (Low, Medium, High)
def categorize_efficiency(score):
    if score < 40:
        return 0  # Low
    elif 40 <= score <= 70:
        return 1  # Medium
    else:
        return 2  # High

# Train fuzzy decision tree
model = FuzzyDecisionTree()
X = df[['ENERGY_CONSUMPTION_PER_SQFT', 'Pct_INCOME_MORE_THAN_150K', 'CLIMATE_Cold', 'CLIMATE_Hot-Humid', 'CLIMATE_Mixed-Humid']]
y = df['EFFICIENCY_SCORE'].apply(categorize_efficiency)
model.fit(X, y)

# Streamlit UI
st.title("🏡 Household Energy Efficiency Predictor")

energy = st.slider("🔋 Energy Consumption per SqFt", 20, 75, 40)
income = st.slider("💰 Percentage Income > 150K", 5, 35, 15)
climate = st.selectbox("🌡️ Climate Type", ["Cold", "Hot-Humid", "Mixed-Humid"])

# Set climate flags
climate_flags = {
    "CLIMATE_Cold": int(climate == "Cold"),
    "CLIMATE_Hot-Humid": int(climate == "Hot-Humid"),
    "CLIMATE_Mixed-Humid": int(climate == "Mixed-Humid")
}

# Predict
input_data = {
    "ENERGY_CONSUMPTION_PER_SQFT": energy,
    "Pct_INCOME_MORE_THAN_150K": income,
    **climate_flags
}
input_df = pd.DataFrame([input_data])

prediction = model.predict(input_df)[0]

# Decode class back to label
label_map = {0: "Low Efficiency", 1: "Medium Efficiency", 2: "High Efficiency"}
predicted_label = label_map.get(prediction, "Unknown")

# Display result
st.subheader(f"🔮 Predicted Efficiency: {predicted_label}")
