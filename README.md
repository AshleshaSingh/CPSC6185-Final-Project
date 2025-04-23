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
├── data/raw/         # Original RECS 2020 CSV files
├── data/processed/   # Merged and cleaned dataset 2020
├── notebooks/        # Data exploration and modeling notebooks
├── app/              # Streamlit app script (app.py)
├── models/           # Saved model files
├── reports/          # Final report and presentation
├── utils/            # Helper functions and utilities
├── tests/            # Unit tests
├── requirements.txt  # Project dependencies
├── README.md         # Project overview
├── .gitignore        # Git ignore rules
└── .github/          # GitHub Actions workflows
```

##  Installation

```bash
pip install -r requirements.txt
```

##  Usage

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

