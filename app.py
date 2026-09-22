import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up the webpage
st.set_page_config(page_title="AI Job Salary Predictor", page_icon="💰", layout="centered")

st.title("🤖 AI Job Salary Predictor")
st.write("Enter the details of the job to predict the estimated salary.")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('best_salary_model.pkl')

model = load_model()

# Create the form for user input
with st.form("prediction_form"):
    st.header("Candidate & Job Details")
    
    col1, col2 = st.columns(2)
    with col1:
        experience_level = st.selectbox("Experience Level", ['EN', 'MI', 'SE', 'EX'], index=1)
        company_size = st.selectbox("Company Size", ['S', 'M', 'L'], index=1)
        education_required = st.selectbox("Education Level", ['Associate', 'Bachelor', 'Master', 'PhD'], index=1)
        job_title = st.selectbox("Job Title", ['AI Research Scientist', 'AI Software Engineer', 'AI Specialist', 'NLP Engineer', 'Data Analyst', 'Machine Learning Engineer', 'Data Engineer'])
    
    with col2:
        industry = st.selectbox("Industry", ['Automotive', 'Media', 'Education', 'Consulting', 'Healthcare', 'Technology', 'Finance'])
        employment_type = st.selectbox("Employment Type", ['FT', 'PT', 'CT', 'FL'])
        company_location = st.selectbox("Company Location", ['United States', 'United Kingdom', 'Canada', 'Germany', 'India', 'China'])
        employee_residence = st.selectbox("Employee Residence", ['United States', 'United Kingdom', 'Canada', 'Germany', 'India', 'China'])
    
    st.header("Additional Factors")
    col3, col4 = st.columns(2)
    with col3:
        years_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=3)
        remote_ratio = st.selectbox("Remote Work Ratio (%)", [0, 50, 100], index=1)
    with col4:
        job_description_length = st.number_input("Job Description Length", min_value=100, max_value=5000, value=1500)
        benefits_score = st.slider("Benefits Score (1-10)", min_value=1.0, max_value=10.0, step=0.1, value=7.5)

    st.write("Select Required Skills:")
    col5, col6, col7, col8 = st.columns(4)
    has_python = col5.checkbox("Python", value=True)
    has_pytorch = col6.checkbox("PyTorch")
    has_aws = col7.checkbox("AWS")
    has_docker = col8.checkbox("Docker")
    has_kubernetes = col5.checkbox("Kubernetes")
    
    submit_button = st.form_submit_button(label="Predict Salary")

if submit_button:
    # Compile the inputs into a dataframe that matches the model's training data
    input_data = pd.DataFrame({
        'experience_level': [experience_level],
        'company_size': [company_size],
        'education_required': [education_required],
        'job_title': [job_title],
        'salary_currency': ["USD"],
        'employment_type': [employment_type],
        'company_location': [company_location],
        'employee_residence': [employee_residence],
        'industry': [industry],
        'years_experience': [years_experience],
        'remote_ratio': [remote_ratio],
        'job_description_length': [job_description_length],
        'benefits_score': [benefits_score],
        'days_open': [30],
        'num_skills': [sum([has_python, has_pytorch, has_aws, has_docker, has_kubernetes])],
        'has_python': [int(has_python)],
        'has_pytorch': [int(has_pytorch)],
        'has_aws': [int(has_aws)],
        'has_docker': [int(has_docker)],
        'has_kubernetes': [int(has_kubernetes)]
    })
    
    try:
        # If your model was trained on log(y), it outputs the correct dollar amount if you used TransformedTargetRegressor. 
        # If not, you may need to wrap it in np.exp() like: prediction = np.exp(model.predict(input_data)[0])
        prediction = model.predict(input_data)[0] 
        st.success(f"### Estimated Salary: ${prediction:,.2f} USD")
        st.balloons()
    except Exception as e:
        st.error(f"Error making prediction: Ensure your input matches the training data format. Details: {e}")