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
💻 Local Setup and Running the App
Follow these steps to run the Streamlit app on your local machine:

1. Clone the Repository
Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create and Activate a Virtual Environment
On macOS/Linux:

Bash
python3 -m venv venv
source venv/bin/activate
On Windows:

Bash
python -m venv venv
venv\Scripts\activate

3. Install Required Dependencies
Create a requirements.txt file (if you haven't already) containing:

Plaintext
streamlit
pandas
numpy
scikit-learn
joblib
Then install them using:

Bash
pip install -r requirements.txt

4. Run the Streamlit Application
Bash
streamlit run app.py
The app will automatically open in your default browser at http://localhost:8501.

📊 Dataset Overview
The underlying dataset (ai_job_dataset.csv) covers various feature categories used in model training:

Target Variable: Estimated Salary (USD)

Key Numeric Features: Candidate Years of Experience, Remote Work Ratio (%), Benefits Score (1-10), Job Description Length, Days Job Open, and Required Skills Count.

Key Categorical Features: Job Title, Industry, Company Location, Employee Residence, Education Level, Employment Type, Company Size, and Experience Level.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
