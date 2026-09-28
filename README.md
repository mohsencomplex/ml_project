# ML Project 01 - Student Performance Prediction

A complete Machine Learning classification project built with Python, scikit-learn, and Streamlit.

This project predicts whether a student is likely to **pass or fail** based on academic, family, and social factors.

The project demonstrates a complete Machine Learning workflow, from data preparation and exploratory analysis to model training, hyperparameter tuning, evaluation, model saving, and deployment through an interactive Streamlit application.

---

## Project Overview

The project implements the following workflow:

- Load the UCI Student Performance dataset
- Explore and analyze the dataset
- Prepare the target variable
- Remove features that could cause target leakage
- Split the data into training and testing sets
- Preprocess numerical and categorical features
- Train multiple Machine Learning models
- Compare model performance
- Perform hyperparameter tuning
- Evaluate the final model
- Save the trained model
- Make predictions on new student data
- Build an interactive Streamlit application

---

## Dataset

This project uses the **Student Performance Dataset** from the UCI Machine Learning Repository.

The project uses the Mathematics dataset (`student-mat.csv`).

| Property | Value |
|---|---:|
| Students | 395 |
| Original Features | 33 |
| Prediction Features | 30 |
| Target | Pass / Fail |
| Training Data | 80% |
| Testing Data | 20% |

### Target Definition

The original target variable is `G3`, representing the student's final grade.

For this project, the target is converted into a binary classification problem:

```text
G3 >= 10  →  Pass
G3 < 10   →  Fail
```

This threshold is a project design choice used to create the binary classification task.

### Important: Target Leakage

The original dataset contains `G1` and `G2`, which are earlier-period grades and are strongly related to the final grade `G3`.

To make the prediction more meaningful as an early prediction task, `G1` and `G2` are intentionally excluded from the model features.

---

## Features

The model uses academic, demographic, family, and social information such as:

- School
- Sex
- Age
- Address
- Family size
- Parent status
- Mother education
- Father education
- Mother job
- Father job
- Reason for choosing school
- Guardian
- Travel time
- Study time
- Previous failures
- School support
- Family support
- Paid classes
- Extra activities
- Nursery education
- Higher education intention
- Internet access
- Romantic relationship
- Family relationship quality
- Free time
- Going out
- Workday alcohol consumption
- Weekend alcohol consumption
- Health
- Absences

---

## Machine Learning Workflow

```text
UCI Student Performance Dataset
              |
              v
        Data Exploration
              |
              v
       Target Preparation
              |
              v
     Remove G1 / G2 Leakage
              |
              v
       Train / Test Split
              |
              v
      Feature Preprocessing
              |
              v
    +---------+----------+
    |         |          |
    v         v          v
Logistic   Random     Gradient
Regression Forest     Boosting
    |         |          |
    +---------+----------+
              |
              v
      Model Comparison
              |
              v
   Random Forest Tuning
              |
              v
       Final Evaluation
              |
              v
       Save Model
              |
              v
      Streamlit App
```

---

## Models

Three classification models were evaluated:

### Logistic Regression

Used as a simple linear baseline model.

### Random Forest

An ensemble learning algorithm based on multiple decision trees.

### Gradient Boosting

An ensemble method that builds models sequentially to improve prediction performance.

---

## Hyperparameter Tuning

Randomized Search with 5-fold cross-validation was used to tune the Random Forest model.

The search evaluated **30 parameter combinations**, resulting in **150 model fits**.

The best configuration included:

```text
n_estimators = 500
max_depth = 15
min_samples_leaf = 4
min_samples_split = 2
max_features = None
class_weight = None
```

The best cross-validation F1 score was:

**81.92%**

The final model is saved as a complete scikit-learn pipeline containing both preprocessing and the trained Random Forest classifier.

---

## Model Evaluation

The final tuned model was evaluated on a completely held-out test set.

| Metric | Test Score |
|---|---:|
| Accuracy | 65.82% |
| Precision | 69.70% |
| Recall | 86.79% |
| F1 Score | 77.31% |

### Confusion Matrix

```text
                 Predicted
              Fail     Pass

Actual Fail      6       20
Actual Pass      7       46
```

The test set contains 79 students.

The model's performance should be interpreted in the context of the relatively small dataset and the binary target definition used in this project.

---

## Prediction

The project includes an interactive prediction script.

Run:

```powershell
python src\predict.py
```

The script asks for student information and loads the trained model to generate a prediction.

Example result:

```text
==================================================
PREDICTION RESULT
==================================================

Prediction: FAIL
Fail probability: 71.86%
Pass probability: 28.14%

Prediction completed successfully!
```

---

## Streamlit Application

The project includes an interactive web application built with Streamlit.

The application allows users to:

- Enter student information
- Enter academic information
- Enter social and lifestyle information
- Generate a prediction
- View Pass probability
- View Fail probability
- See the final prediction

Run the application:

```powershell
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## Project Structure

```text
ml_project/
│
├── data/
│   └── student_performance/
│
├── models/
│   └── student_performance_model.joblib
│
├── src/
│   ├── main.py
│   ├── student_performance.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Important

The dataset and trained model are excluded from Git using `.gitignore`.

The trained model can be regenerated by running:

```powershell
python src\student_performance.py
```

---

## Installation

Clone the repository:

```powershell
git clone https://github.com/mohsencomplex/ml_project.git
cd ml_project
```

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## Train the Model

Run:

```powershell
python src\student_performance.py
```

The script will:

1. Load the dataset
2. Prepare the target
3. Split the data
4. Preprocess categorical and numerical features
5. Train the baseline models
6. Tune the Random Forest model
7. Evaluate the final model
8. Save the trained pipeline

The trained model will be saved to:

```text
models/student_performance_model.joblib
```

---

## Make a Prediction

Run:

```powershell
python src\predict.py
```

Enter the requested student information.

The script will display:

```text
Prediction: PASS / FAIL
Fail probability: XX.XX%
Pass probability: XX.XX%
```

---

## Run the Web Application

After training the model:

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Technologies

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Joblib
- Streamlit
- Git
- GitHub

---

## Machine Learning Concepts Practiced

This project was created to practice practical Machine Learning concepts including:

- Data preprocessing
- Categorical feature encoding
- Numerical feature handling
- Train/Test splitting
- Stratified sampling
- Classification
- Logistic Regression
- Random Forest
- Gradient Boosting
- Cross-validation
- F1 score
- Precision and Recall
- Confusion Matrix
- Hyperparameter tuning
- RandomizedSearchCV
- Target leakage prevention
- Model persistence with Joblib
- Interactive ML deployment with Streamlit

---

## Future Improvements

Possible future improvements include:

- Feature importance analysis
- More detailed exploratory data analysis
- Model interpretability
- Cross-dataset evaluation
- Improved user interface
- Docker deployment
- Cloud deployment
- Experiment tracking
- Additional Machine Learning models

---

## Author

**Mohsen El-Sharnoby**

AI Student | Machine Learning Enthusiast

GitHub: `https://github.com/mohsencomplex`