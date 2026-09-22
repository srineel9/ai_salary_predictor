import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Job Salary Predictor", page_icon="💰", layout="centered")
st.title("🤖 AI Job Salary Predictor")
st.write("Enter the details of the job to predict the estimated salary.")

# ----------------------------------------------------------------------------
# Load the trained model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_salary_model.pkl")

model = load_model()

# ----------------------------------------------------------------------------
# Option lists -- pulled directly from the training data so every real
# category is selectable (previously this only had a handful of each).
# ----------------------------------------------------------------------------
JOB_TITLES = [
    'AI Architect', 'AI Consultant', 'AI Product Manager', 'AI Research Scientist',
    'AI Software Engineer', 'AI Specialist', 'Autonomous Systems Engineer',
    'Computer Vision Engineer', 'Data Analyst', 'Data Engineer', 'Data Scientist',
    'Deep Learning Engineer', 'Head of AI', 'ML Ops Engineer',
    'Machine Learning Engineer', 'Machine Learning Researcher', 'NLP Engineer',
    'Principal Data Scientist', 'Research Scientist', 'Robotics Engineer',
]

INDUSTRIES = [
    'Automotive', 'Consulting', 'Education', 'Energy', 'Finance', 'Gaming',
    'Government', 'Healthcare', 'Manufacturing', 'Media', 'Real Estate',
    'Retail', 'Technology', 'Telecommunications', 'Transportation',
]

COUNTRIES = [
    'Australia', 'Austria', 'Canada', 'China', 'Denmark', 'Finland', 'France',
    'Germany', 'India', 'Ireland', 'Israel', 'Japan', 'Netherlands', 'Norway',
    'Singapore', 'South Korea', 'Sweden', 'Switzerland', 'United Kingdom',
    'United States',
]

# All 24 skills that actually appear in the training data's `required_skills` column
ALL_SKILLS = [
    'Python', 'SQL', 'R', 'Java', 'Scala', 'Statistics', 'Mathematics',
    'Deep Learning', 'NLP', 'Computer Vision', 'MLOps', 'TensorFlow', 'PyTorch',
    'Data Visualization', 'Tableau', 'Git', 'Docker', 'Kubernetes', 'Linux',
    'Hadoop', 'Spark', 'AWS', 'Azure', 'GCP',
]

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("prediction_form"):
    st.header("Candidate & Job Details")

    col1, col2 = st.columns(2)
    with col1:
        experience_level = st.selectbox(
            "Experience Level", ['EN', 'MI', 'SE', 'EX'], index=1,
            help="EN = Entry, MI = Mid, SE = Senior, EX = Executive",
        )
        company_size = st.selectbox("Company Size", ['S', 'M', 'L'], index=1)
        education_required = st.selectbox(
            "Education Level", ['Associate', 'Bachelor', 'Master', 'PhD'], index=1
        )
        job_title = st.selectbox("Job Title", JOB_TITLES)

    with col2:
        industry = st.selectbox("Industry", INDUSTRIES)
        employment_type = st.selectbox("Employment Type", ['FT', 'PT', 'CT', 'FL'])
        company_location = st.selectbox("Company Location", COUNTRIES,
                                         index=COUNTRIES.index('United States'))
        employee_residence = st.selectbox("Employee Residence", COUNTRIES,
                                           index=COUNTRIES.index('United States'))

    st.header("Additional Factors")
    col3, col4 = st.columns(2)
    with col3:
        years_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=3)
        remote_ratio = st.selectbox("Remote Work Ratio (%)", [0, 50, 100], index=1)
    with col4:
        job_description_length = st.number_input(
            "Job Description Length", min_value=100, max_value=5000, value=1500
        )
        benefits_score = st.slider(
            "Benefits Score (1-10)", min_value=1.0, max_value=10.0, step=0.1, value=7.5
        )

    st.header("Required Skills")
    st.caption("Select every skill listed in the job posting.")
    selected_skills = []
    skill_cols = st.columns(4)
    for i, skill in enumerate(ALL_SKILLS):
        # Default a couple of common ones to checked just so the form isn't empty
        default_checked = skill in ("Python", "SQL")
        if skill_cols[i % 4].checkbox(skill, value=default_checked, key=f"skill_{skill}"):
            selected_skills.append(skill)

    submit_button = st.form_submit_button(label="Predict Salary")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submit_button:
    if len(selected_skills) == 0:
        st.warning("Select at least one skill for a more accurate prediction.")

    # The model was trained on these exact columns -- num_skills is simply
    # the COUNT of required skills (the model never saw one column per skill,
    # so we only need the count here, not has_python / has_aws / etc.)
    input_data = pd.DataFrame({
        "job_title": [job_title],
        "experience_level": [experience_level],
        "employment_type": [employment_type],
        "company_location": [company_location],
        "company_size": [company_size],
        "employee_residence": [employee_residence],
        "education_required": [education_required],
        "industry": [industry],
        "years_experience": [years_experience],
        "remote_ratio": [remote_ratio],
        "job_description_length": [job_description_length],
        "benefits_score": [benefits_score],
        "num_skills": [len(selected_skills)],
        "days_open": [30],  # typical value; not collected from the user
    })

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"### Estimated Salary: ${prediction:,.2f} USD")
        if selected_skills:
            st.caption(f"Skills counted: {', '.join(selected_skills)}")
        st.balloons()
    except Exception as e:
        st.error(f"Error making prediction: Ensure your input matches the training data format. Details: {e}")
