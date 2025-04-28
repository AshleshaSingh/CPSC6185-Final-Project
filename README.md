# CPSC6185-Final-Project
Final Project for course CPSC6185-Inteligent Systems

# AI-Based Household Energy Efficiency Classifier

A Streamlit app to classify US states (RECS 2020) as High, Moderate, or Low Energy Efficient using Decision Trees and Fuzzy Logic.

##  Features
- Merges and preprocesses RECS 2020 state-level data
- Computes energy efficiency metric (kWh/sqft)
- Classifies efficiency using a Decision Tree
- Refines classifications with a Fuzzy Logic system
- Visualizes predictions, decision paths, and fuzzy scores in an interactive app

##  Repository Structure

```
CPSC6185-Final-Project/
├── data/
│   ├── raw/          # Original RECS 2020 CSV files
│   │   ├── recs_annual_household_energy_consumption_and_expenditure.csv
│   │   └── ... (other raw CSVs)
│   ├── processed/    # Merged and cleaned dataset 2020
│   │   ├── merged_cleaned.csv
│   │   ├── merged_with_efficiency.csv
├── models/          # Saved model files 
│   ├── decision_tree_model.pkl
├── notebooks/        # Data exploration and modeling notebooks
│   ├── preprocessing.ipynb
│   ├── fuzzy_logic.ipynb
│   ├── decision_tree_classifier.ipynb
│   ├── FuzzyDecisionTree.ipynb
├── app/              # Streamlit app script (app.py)
├── utils/            # Helper functions and utilities
├── tests/            # Unit tests
├── requirements.txt  # Project dependencies
├── README.md         # Project overview
├── .gitignore        # Git ignore rules
└── .github/          # GitHub Actions workflows
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AshleshaSingh/CPSC6185-Final-Project.git
   cd CPSC6185-Final-Project
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure raw data files are in `data/raw/` (download from Redivis or EIA).

## Usage

1. **Preprocessing**:
   Run `preprocessing.ipynb` to merge raw CSVs and generate `merged_cleaned.csv`:
   ```bash
   jupyter notebook notebooks/preprocessing.ipynb
   ```

2. **Fuzzy Logic**:
   Run `fuzzy_logic.ipynb` to compute fuzzy scores and generate `merged_with_efficiency.csv`:
   ```bash
   jupyter notebook notebooks/fuzzy_logic.ipynb
   ```

3. **Decision Tree**:
   Run `decision_tree_classifier.ipynb` to train and evaluate the Decision Tree model:
   ```bash
   jupyter notebook notebooks/decision_tree_classifier.ipynb
   ```

4. **Fuzzy Decision Tree**:
   Run `FuzzyDecisionTree.ipynb` to train and evaluate the custom Fuzzy Decision Tree:
   ```bash
   jupyter notebook notebooks/FuzzyDecisionTree.ipynb
   ```

5. **Streamlit App**:
   Run the Streamlit app locally:
   ```bash
   streamlit run app/app.py
   ```

##  Dependencies

- pandas
- numpy
- scikit-learn
- scikit-fuzzy
- matplotlib
- seaborn
- streamlit

##  Authors
- Ashlesha Singh
- Ai Tran
- Fnu Swati
- Mekaila Quarshie 
