import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings if any

st.set_page_config(page_title="Employee Engagement Dashboard", layout="wide")


@st.cache_data
def load_artifacts():
    df = pd.read_csv('processed_employee_data.csv')

    try:
        model = joblib.load('salary_prediction_model.pkl')
        # st.success("Model loaded from pickle.")  # Commented: No messages
    except Exception as e:
        # Silent: Terminal only
        print(f"Model load failed ({e}). Re-training now...")
        # Re-train here (copy from Jupyter Step 6)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        features = ['Engagement_Index',
                    'Experience_Years', 'Performance_Category']
        X = df[features].copy()
        y = df['Salary']

        cat_features = ['Performance_Category']
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(drop='first'), cat_features)],
            remainder='passthrough'
        )

        pipeline = Pipeline(
            [('prep', preprocessor), ('model', LinearRegression())])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)

        # Quick eval
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Re-trained R²: {r2:.3f}")  # Console debug

        # Save for next time
        joblib.dump(pipeline, 'salary_prediction_model.pkl')
        model = pipeline
        # st.success("Model ready!")  # Optional: Uncomment for subtle confirmation

    with open('eda_summary.json', 'r') as f:
        summary = json.load(f)

    # Update summary with fresh R² if re-trained (simple check)
    if 'r2' not in locals():
        r2 = summary.get('model_r2', 0.65)  # Fallback default
    summary['model_r2'] = float(r2)

    return df, model, summary


df, model, summary = load_artifacts()

# Sidebar
st.sidebar.header("🔍 Filters")
depts = st.sidebar.multiselect("Department", sorted(
    df['Department'].unique()), default=df['Department'].unique())
age_range = st.sidebar.slider("Age", df['Age'].min(
), df['Age'].max(), (df['Age'].min(), df['Age'].max()))
cats = st.sidebar.multiselect("Perf Category (for dashboard)", df['Performance_Category'].unique(
), default=df['Performance_Category'].unique())

st.sidebar.header("🔮 Predict Salary")
eng = st.sidebar.slider("Engagement Index", 1.0, 10.0, 7.0)
exp = st.sidebar.slider("Years Exp", 0, int(df['Experience_Years'].max()), 5)
cat_input = st.sidebar.selectbox(
    "Perf Category (for prediction)", df['Performance_Category'].unique())

# Filters
filtered_df = df[(df['Department'].isin(depts)) & (
    df['Age'].between(*age_range)) & (df['Performance_Category'].isin(cats))]

# Predict func


@st.cache_data
def predict(eng, exp, cat):
    input_df = pd.DataFrame({'Engagement_Index': [eng], 'Experience_Years': [
                            exp], 'Performance_Category': [cat]})
    return model.predict(input_df)[0]


pred_salary = predict(eng, exp, cat_input)

# Layout
st.title("🚀 Employee Engagement & Performance Dashboard")
st.markdown("Jupyter-powered insights: Explore trends, predict salaries.")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Avg Perf", f"{filtered_df['Performance_Score'].mean():.2f}", f"{summary['avg_performance']:.2f}")
col2.metric("Avg Salary",
            f"${filtered_df['Salary'].mean():,.0f}", f"${summary['avg_salary']:,.0f}")
col3.metric("% High Perf",
            f"{(len(filtered_df[filtered_df['Performance_Category'] == 'High']) / len(filtered_df) * 100):.1f}%", f"{summary['high_performers_pct']:.1f}%")
col4.metric("Model R²", f"{summary['model_r2']:.3f}")

# Prediction
st.subheader("💰 Salary Prediction")
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.write("**Profile:**")
    st.write(f"- Engagement: {eng}")
    st.write(f"- Experience: {exp} yrs")
    st.write(f"- Category: {cat_input}")
with col_p2:
    st.success(f"**Predicted: ${pred_salary:,.0f}")
    st.caption("Linear Regression based on data.")

# Viz Row 1
st.subheader("📊 Overview")
col_v1, col_v2 = st.columns(2)
with col_v1:
    fig_h = px.histogram(filtered_df, x='Performance_Score',
                         nbins=20, title="Performance Distribution")
    st.plotly_chart(fig_h, use_container_width=True)
with col_v2:
    dept_sal = filtered_df.groupby('Department')['Salary'].mean().reset_index()
    fig_b = px.bar(dept_sal, x='Department',
                   y='Salary', title="Avg Salary/Dept")
    st.plotly_chart(fig_b, use_container_width=True)


# Insights
st.subheader("💡 Insights")
insights = pd.DataFrame({
    'Metric': ['Avg Perf', 'Avg Salary', 'High %', 'Salary-Perf Corr'],
    'Value': [f"{filtered_df['Performance_Score'].mean():.2f}", f"${filtered_df['Salary'].mean():,.0f}",
              f"{(len(filtered_df[filtered_df['Performance_Category'] == 'High']) / len(filtered_df) * 100):.1f}%",
              f"{filtered_df[['Performance_Score', 'Salary']].corr().iloc[0, 1]:.2f}"],
    'Rec': ['Target low depts', 'Check equity', 'Uplift mediums', 'Reward high-perf']
})
st.table(insights)

st.markdown("---")
st.caption("Update: Re-run Jupyter, push artifacts to Git → auto-refresh on host.")
