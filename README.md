# 🤖 AI & Data Job Salary Predictor

A End-to-End Machine Learning web application designed to predict estimated salaries for Artificial Intelligence and Data Science professionals based on industry parameters, job roles, candidate experience, education, and specific technical skill sets. 

The interactive web dashboard is built with **Streamlit** and deployed on **Streamlit Cloud**.

---

## 🚀 Live Demo

🔗 https://aisalarypredictor.streamlit.app/

---

## 📌 Features

- **Dynamic Interactive UI:** Built using Streamlit for intuitive inputs across candidate profiles, company details, and compensation factors.
- **Comprehensive Skill Selection:** Dynamically supports 24 essential AI/Data skills (e.g., Python, SQL, PyTorch, TensorFlow, MLOps, AWS, GCP, Docker, Kubernetes).
- **Comprehensive Categorical Coverage:** Offers selection across 20 distinct AI & Data job titles, 15 industries, 20 candidate/company countries, experience levels, and company sizes.
- **Pipeline Architecture:** Leverages Scikit-Learn pipelines combining custom column preprocessors (`StandardScaler`, `OneHotEncoder`) with gradient boosting regressor trees for precise salary estimation.

---

## 📁 Repository Structure

```text
├── app.py                   # Streamlit web application script
├── best_salary_model.pkl    # Pre-trained Scikit-Learn pipeline model
├── ai_job_dataset.csv       # Dataset used for training and testing
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation

🛠️ Tech Stack & Dependencies
Language: Python

Web Framework: Streamlit

Machine Learning & Pipeline: Scikit-Learn

Data Manipulation: Pandas, NumPy

Model Serialization: Joblib

