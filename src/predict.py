import joblib
import pandas as pd

MODEL_PATH = "models/student_performance_model.joblib"

model = joblib.load(MODEL_PATH)

print("=" * 50)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 50)

student = {
    "school": input("School (GP/MS): "),
    "sex": input("Sex (M/F): "),
    "age": int(input("Age: ")),
    "address": input("Address (U/R): "),
    "famsize": input("Family size (GT3/LE3): "),
    "Pstatus": input("Parent status (T/A): "),
    "Medu": int(input("Mother education (0-4): ")),
    "Fedu": int(input("Father education (0-4): ")),
    "Mjob": input("Mother job (teacher/health/services/at_home/other): "),
    "Fjob": input("Father job (teacher/health/services/at_home/other): "),
    "reason": input("Reason (course/home/reputation/other): "),
    "guardian": input("Guardian (mother/father/other): "),
    "traveltime": int(input("Travel time (1-4): ")),
    "studytime": int(input("Study time (1-4): ")),
    "failures": int(input("Past failures (0-4): ")),
    "schoolsup": input("School support (yes/no): "),
    "famsup": input("Family support (yes/no): "),
    "paid": input("Extra paid classes (yes/no): "),
    "activities": input("Extra activities (yes/no): "),
    "nursery": input("Attended nursery (yes/no): "),
    "higher": input("Wants higher education (yes/no): "),
    "internet": input("Internet access (yes/no): "),
    "romantic": input("Romantic relationship (yes/no): "),
    "famrel": int(input("Family relationship quality (1-5): ")),
    "freetime": int(input("Free time (1-5): ")),
    "goout": int(input("Going out frequency (1-5): ")),
    "Dalc": int(input("Workday alcohol consumption (1-5): ")),
    "Walc": int(input("Weekend alcohol consumption (1-5): ")),
    "health": int(input("Current health status (1-5): ")),
    "absences": int(input("Number of absences: "))
}

student_df = pd.DataFrame([student])

prediction = model.predict(student_df)[0]
probability = model.predict_proba(student_df)[0]

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

print("\nPrediction:", "PASS" if prediction == 1 else "FAIL")
print(f"Fail probability: {probability[0]:.2%}")
print(f"Pass probability: {probability[1]:.2%}")

print("\nPrediction completed successfully!")