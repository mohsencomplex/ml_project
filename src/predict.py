import joblib

model = joblib.load("iris_model.joblib")

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print(f"Predicted class: {prediction[0]}")