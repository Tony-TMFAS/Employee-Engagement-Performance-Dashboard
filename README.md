## Employee Engagement & Performance Dashboard

This project implements an interactive HR analytics dashboard for analyzing employee performance, engagement, and salary trends using a synthetic dataset. The dashboard enables users to filter data by department, age, and performance category, visualize key metrics, and predict salaries for hypothetical employee profiles using a trained Linear Regression model.
The dataset is the Employee Performance & Salary (Synthetic Dataset) (~1,020 rows), featuring columns like Employee_ID, Age, Gender, Department, Experience_Years, Performance_Score (1-10 scale as engagement proxy), and Salary. Performance is categorized as Low/Medium/High, and an Engagement_Index is engineered as a normalized blend (70% performance score + 30% experience).

## Key capabilities:

Exploratory views of distributions and correlations.
Dynamic filtering and KPI monitoring.
Model-based salary predictions (R² ≈ 0.65-0.75 on test data).

Live Demo: https://employee-engagement-performance-dashboard.streamlit.app/

## Table of Contents

Features
Technical Stack
Project Structure
Usage
Model Details
Deployment


## Features

Interactive Filters: Multiselect for departments and performance categories; slider for age range. Updates all views in real-time.
Key Performance Indicators (KPIs): Displays averages for performance score, salary, % high performers, and model R².
Salary Prediction Tool: Input engagement index, experience years, and performance category to forecast salary using a pre-trained model.
Visualizations:

Histogram: Performance score distribution.
Bar Chart: Average salary by department.
Box Plot: Prediction residuals (errors) by department and category.


Insights Table: Summary metrics with actionable recommendations.
Modular Design: Data preprocessing and modeling in Jupyter; lightweight Streamlit for display.

Technical Stack

Data Processing & Modeling: Python 3.11+ with Pandas, Scikit-learn (Linear Regression, ColumnTransformer, OneHotEncoder).
Visualization: Plotly (interactive charts: histograms, bars, scatters, boxes).
Dashboard Framework: Streamlit (filters, metrics, layout).
Utilities: Joblib (model serialization), JSON (summary stats).
Environment: Jupyter Notebook for development; no external APIs.

Project Structure
textemployee-engagement-performance-dashboard/
├── dashboard.py              # Streamlit app script
├── requirements.txt          # Dependencies
├── runtime.txt               # Python version pin (3.11.9)
├── processed_employee_data.csv  # Cleaned/enhanced dataset (with engineered features & predictions)
├── salary_prediction_model.pkl  # Trained Linear Regression pipeline
├── eda_summary.json         # Pre-computed stats (e.g., averages, R²)
├── README.md                # This file
└── jupyter/                 # (Optional: Development notebook)
    └── Preprocessing_Analysis_Modelling.ipynb  # Full analysis workflow


## For Jupyter development:

Install extras: pip install jupyter ipywidgets seaborn matplotlib.
Run: jupyter notebook jupyter/employee_dashboard.ipynb.

## Usage

Launch the Dashboard: Open the app in your browser.
Apply Filters: Use sidebar multiselects/sliders to narrow data (e.g., IT department, ages 25-35, High/Medium performance).
View KPIs & Charts: Metrics update automatically; explore distributions and dept comparisons.
Predict Salaries: In the sidebar, adjust sliders/selectbox for a profile → See predicted salary (e.g., High performer with 8.5 engagement and 10 years exp ≈ $85,000).
Insights: Review the table for data-driven recs (e.g., "Target low depts").

## Example Workflow:

Filter to Sales dept → Observe higher salary variance.
Predict for a new hire: Medium category, 6.0 engagement, 3 years → ~$55,000.

## Model Details

Objective: Predict Salary from Engagement_Index, Experience_Years, and Performance_Category.
Approach: Linear Regression via Scikit-learn Pipeline (OneHotEncoder for categorical features).
Training: 80/20 train-test split (random_state=42); features engineered in Jupyter (imputation, categorization, indexing).
Performance: R² ≈ 0.841; MAE ≈ $4,450 (on test set). Residuals visualized for bias checks (e.g., over-prediction in certain depts).
Re-training: If pickle fails (e.g., version mismatch), the app auto-retrains on load using the CSV.

## Deployment

Platform: Streamlit Cloud (free tier).
Steps:

Push to GitHub (public repo).
Connect repo in Streamlit Cloud → Set main file: dashboard.py.
Add runtime.txt for Python 3.11 (avoids 3.13 issues).
Auto-deploys on push; reboot via "Manage app" for updates.


Live URL: https://employee-engagement-performance-dashboard.streamlit.app/
Notes: Artifacts (CSV/PKL) must be <100MB; app re-trains model if needed.

ta.
Built with Streamlit, Plotly, and Scikit-learn—thanks to their open-source communities.
