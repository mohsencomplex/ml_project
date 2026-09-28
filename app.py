import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/student_performance_model.joblib"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()

st.title("🎓 Student Performance Predictor")
st.write(
    "Predict whether a student is likely to pass or fail "
    "based on academic, family, and social factors."
)

st.info(
    "This model was trained on the UCI Student Performance dataset. "
    "G1 and G2 are intentionally excluded from prediction."
)

st.divider()

st.header("👤 Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["M", "F"])
    age = st.number_input("Age", min_value=15, max_value=22, value=17)
    address = st.selectbox("Address", ["U", "R"])

with col2:
    famsize = st.selectbox("Family Size", ["GT3", "LE3"])
    pstatus = st.selectbox("Parent Status", ["T", "A"])
    medu = st.selectbox("Mother Education", [0, 1, 2, 3, 4], index=2)
    fedu = st.selectbox("Father Education", [0, 1, 2, 3, 4], index=2)

with col3:
    mjob = st.selectbox(
        "Mother Job",
        ["teacher", "health", "services", "at_home", "other"]
    )
    fjob = st.selectbox(
        "Father Job",
        ["teacher", "health", "services", "at_home", "other"]
    )
    reason = st.selectbox(
        "Reason for Choosing School",
        ["course", "home", "reputation", "other"]
    )
    guardian = st.selectbox(
        "Guardian",
        ["mother", "father", "other"]
    )

st.divider()

st.header("📚 Academic Information")

col1, col2, col3 = st.columns(3)

with col1:
    traveltime = st.slider("Travel Time", 1, 4, 2)
    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.number_input(
        "Past Failures",
        min_value=0,
        max_value=4,
        value=0
    )

with col2:
    schoolsup = st.selectbox("School Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])

with col3:
    activities = st.selectbox("Extra Activities", ["yes", "no"])
    nursery = st.selectbox("Attended Nursery", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])

st.divider()

st.header("🌐 Social & Lifestyle Information")

col1, col2, col3 = st.columns(3)

with col1:
    internet = st.selectbox("Internet Access", ["yes", "no"])
    romantic = st.selectbox("Romantic Relationship", ["yes", "no"])
    famrel = st.slider("Family Relationship Quality", 1, 5, 4)

with col2:
    freetime = st.slider("Free Time", 1, 5, 3)
    goout = st.slider("Going Out", 1, 5, 3)
    dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)

with col3:
    walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)
    health = st.slider("Current Health Status", 1, 5, 3)
    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=4
    )

st.divider()

student = pd.DataFrame([{
    "school": school,
    "sex": sex,
    "age": age,
    "address": address,
    "famsize": famsize,
    "Pstatus": pstatus,
    "Medu": medu,
    "Fedu": fedu,
    "Mjob": mjob,
    "Fjob": fjob,
    "reason": reason,
    "guardian": guardian,
    "traveltime": traveltime,
    "studytime": studytime,
    "failures": failures,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "paid": paid,
    "activities": activities,
    "nursery": nursery,
    "higher": higher,
    "internet": internet,
    "romantic": romantic,
    "famrel": famrel,
    "freetime": freetime,
    "goout": goout,
    "Dalc": dalc,
    "Walc": walc,
    "health": health,
    "absences": absences
}])

if st.button(
    "🔮 Predict Student Performance",
    type="primary",
    use_container_width=True
):
    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]

    fail_probability = probabilities[0]
    pass_probability = probabilities[1]

    st.divider()
    st.header("📊 Prediction Result")

    if prediction == 1:
        st.success("### ✅ PASS")
    else:
        st.error("### ❌ FAIL")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pass Probability",
            f"{pass_probability:.2%}"
        )
        st.progress(float(pass_probability))

    with col2:
        st.metric(
            "Fail Probability",
            f"{fail_probability:.2%}"
        )
        st.progress(float(fail_probability))

    if prediction == 1:
        st.success(
            "The model predicts that this student is likely to pass."
        )
    else:
        st.warning(
            "The model predicts that this student is likely to fail."
        )

st.divider()

st.caption(
    "Machine Learning Project — Student Performance Prediction"
)