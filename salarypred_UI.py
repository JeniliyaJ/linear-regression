import streamlit as st
import pandas as pd
import joblib 

model=joblib.load('employee_salary_model.pkl')
scale=joblib.load('employee_salary_scaler.pkl')

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="wide")
st.title("💼 Employee Salary Prediction System")
st.markdown("Use this AI-powered tool to predict estimated monthly salary...")
st.divider()

with st.sidebar:
    st.header("ℹ️ About App")
    st.info("This model uses **SelectKBest** features...")
    st.write("---")
    st.write("Developed for HR Analytics.")

# split the page into two columns
col1, col2 = st.columns(2)

with col1:
    job_level = st.selectbox("Job Level", [1,2,3,4,5],index=1)
    total_working_years = st.slider("Total Working Years",0,40,5)
    years_at_company=st.slider("Years At Company", 0, 30, 2)

with col2:
    age=st.number_input("Employee age",min_value=18, max_value=60, value=30)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    
if st.button("🚀 Predict Monthly Income", use_container_width=True):
    input_feature=pd.DataFrame({"JobLevel": [job_level],
                                "TotalWorkingYears":[total_working_years],
                                "YearsAtCompany":[years_at_company],
                                "Age":[age],
                                "YearsInCurrentRole":[years_since_last_promotion]})

    input_feature_scaled=scale.transform(input_feature)
    prediction=model.predict(input_feature_scaled)[0]

    st.success("✅ Prediction Successful!")
    st.metric(label="Estimated Monthly Income", value=f"₹ {prediction:.2f}")
    