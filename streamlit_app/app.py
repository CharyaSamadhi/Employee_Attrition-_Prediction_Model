import streamlit as st
import pandas as pd
import joblib


# PAGE CONFIG

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="👨‍💼", layout="wide")


# Load trained pipeline silently & show centered loading screen

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return joblib.load("models/final_model.pkl")

# Create a centered placeholder while loading
placeholder = st.empty()
placeholder.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <div style="text-align: center; font-size: 24px; color: white;">
            ⏳ <b>Loading model... please wait</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model
pipeline = load_pipeline()

# Clear the loading message after model loads
placeholder.empty()


# HEADER

st.title("👨‍💼 Employee Attrition Prediction App")
st.markdown("Provide employee details to predict if they are likely to **resign or stay**.")


# Fixed categorical options

departments = ['IT', 'Finance', 'Customer Support', 'Engineering', 'Marketing', 'HR', 'Operations', 'Sales', 'Legal']
genders = ['Male', 'Female', 'Other']
job_titles = ['Specialist', 'Developer', 'Analyst', 'Manager', 'Technician', 'Engineer', 'Consultant']
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']


# Reset function

def reset_form():
    """Completely clears all form fields."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Form has been reset!")
    st.rerun()


# SESSION DEFAULTS

defaults = {
    "age": 22, "gender": "", "education": "",
    "satisfaction": 3.0, "department": "", "job_role": "",
    "salary": 1000, "years_at_company": 1,
    "work_hours": 40, "overtime": 0,
    "team_size": 0, "projects_handled": 0,
    "promotions": 0, "training_hours": 0, "sick_days": 0,
    "performance": 3, "remote_work": "Never (0%)"
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# INPUT FORM

with st.form("employee_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    # LEFT COLUMN 
    with col1:
        st.subheader("👤 Personal Information")
        age = st.number_input("Age", min_value=22, max_value=60, key="age")
        gender = st.selectbox("Gender", [""] + genders, key="gender")
        education = st.selectbox("Education Level", [""] + education_levels, key="education")
        satisfaction = st.number_input(
            "Employee Satisfaction Score (0.00 - 5.00)",
            min_value=0.00, max_value=5.00, step=0.01, format="%.2f", key="satisfaction"
        )

        st.subheader("💼 Work Information")
        department = st.selectbox("Department", [""] + departments, key="department")
        job_role = st.selectbox("Job Title", [""] + job_titles, key="job_role")
        salary = st.number_input("Monthly Salary (USD)", min_value=1000, max_value=200000, step=500, key="salary")
        years_at_company = st.slider("Years at Company", 0, 40, key="years_at_company")
        work_hours = st.number_input("Work Hours per Week", min_value=20, max_value=80, key="work_hours")
        overtime = st.number_input("Overtime Hours (last year)", min_value=0, max_value=500, key="overtime")

    # RIGHT COLUMN 
    with col2:
        st.subheader("📊 Team & Performance")
        team_size = st.number_input("Team Size", min_value=0, max_value=30, key="team_size")
        projects_handled = st.number_input("Projects Handled", min_value=0, max_value=50, key="projects_handled")
        promotions = st.number_input("Number of Promotions", min_value=0, max_value=10, key="promotions")
        training_hours = st.number_input("Training Hours (per year)", min_value=0, max_value=500, key="training_hours")
        sick_days = st.number_input("Sick Days", min_value=0, max_value=365, key="sick_days")
        performance = st.selectbox("Performance Score (1-5)", [1, 2, 3, 4, 5], key="performance")
        remote_work = st.selectbox(
            "Remote Work Frequency",
            ["Never (0%)", "Occasionally (25%)", "Half Time (50%)", "Mostly (75%)", "Always (100%)"],
            key="remote_work"
        )

    # --- BUTTONS ---
    colb1, colb2 = st.columns([0.5, 0.5])
    with colb1:
        submitted = st.form_submit_button("🔮 Predict Attrition", use_container_width=True)
    with colb2:
        reset_clicked = st.form_submit_button("🔄 Reset Form", use_container_width=True)

# ===========================
# RESET ACTION
# ===========================
if reset_clicked:
    reset_form()

# ===========================
# Prediction Logic
# ===========================
remote_map = {
    "Never (0%)": 0, "Occasionally (25%)": 25, "Half Time (50%)": 50,
    "Mostly (75%)": 75, "Always (100%)": 100
}
remote_value = remote_map.get(remote_work)

if submitted:
    if (
        gender == "" or department == "" or job_role == "" or education == "" or
        salary <= 0 or remote_value is None
    ):
        st.error("⚠️ Please fill **all required fields** before predicting.")
    else:
        X_input = pd.DataFrame([{
            "Age": age, "Gender": gender, "Education_Level": education,
            "Employee_Satisfaction_Score": satisfaction, "Department": department,
            "Job_Title": job_role, "Monthly_Salary": salary,
            "Years_At_Company": years_at_company, "Work_Hours_Per_Week": work_hours,
            "Overtime_Hours": overtime, "Team_Size": team_size,
            "Projects_Handled": projects_handled, "Promotions": promotions,
            "Training_Hours": training_hours, "Sick_Days": sick_days,
            "Performance_Score": performance, "Remote_Work_Frequency": remote_value
        }])

        proba = pipeline.predict_proba(X_input)[0][1]
        THRESHOLD = 0.027
        prediction = int(proba >= THRESHOLD)

        # --- Display Result ---
        st.subheader("🔮 Prediction Result")
        if prediction == 1:
            st.markdown(
                """
                <div style="padding:20px; background-color:black; border-radius:10px; text-align:center; font-size:22px; font-weight:bold;">
                ⚠️ Employee is likely to <span style="color:red;">RESIGN</span>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="padding:20px; background-color:black; border-radius:10px; text-align:center; font-size:22px; font-weight:bold;">
                ✅ Employee is likely to <span style="color:green;">STAY</span>
                </div>
                """, unsafe_allow_html=True
            )
