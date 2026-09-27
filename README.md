# ML Project 01 - Iris Classification



A complete Machine Learning classification project built with Python and scikit-learn.



This project demonstrates a complete Machine Learning workflow, from preparing the data and training the model to evaluating its performance and making new predictions.



---



## Project Overview



The project implements the following workflow:



- Load the Iris dataset

- Split data into training and testing sets

- Train a Decision Tree classifier

- Evaluate model performance

- Save the trained model

- Load the saved model

- Make new predictions



---



## Dataset



This project uses the built-in Iris dataset provided by scikit-learn.



| Property | Value |

|---|---:|

| Samples | 150 |

| Features | 4 |

| Classes | 3 |

| Training Data | 80% |

| Testing Data | 20% |



### Features



- Sepal Length

- Sepal Width

- Petal Length

- Petal Width



---



## Machine Learning Model



The project uses a **Decision Tree Classifier**.



The data is divided into training and testing sets. The model learns from the training data and is then evaluated on previously unseen test data.



```text

Dataset

   |

   v

Train / Test Split

   |

   +-- Training Data --> Decision Tree --> Trained Model

   |

   +-- Test Data -----------------------> Evaluation

                                             |

                                             v

                                          Accuracy

```



---



## Results



The trained model achieved:



**93.33% Accuracy**



The model was evaluated using `accuracy_score` from scikit-learn.



---



## Prediction



After training, the model is saved using Joblib.



The `predict.py` script loads the saved model and uses new flower measurements to generate a prediction.



Example input:



```text

[5.1, 3.5, 1.4, 0.2]

```



Example prediction:



```text

Predicted class: 0

```



---



## Project Structure



```text

ML Project 01/

|

+-- data/

|

+-- src/

|   +-- main.py

|   +-- predict.py

|

+-- .gitignore

+-- requirements.txt

+-- README.md

```



The trained `.joblib` model is generated locally and excluded from Git using `.gitignore`.



---



## Installation



Clone the repository:



```bash

git clone https://github.com/mohsencomplex/ml_project.git

cd ml_project

```



Create a Python 3.12 virtual environment:



```powershell

py -3.12 -m venv .venv

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

python src\main.py

```



Expected output:



```text

Accuracy: 93.33%

Model saved successfully!

```



---



## Make a Prediction



Run:



```powershell

python src\predict.py

```



Example output:



```text

Predicted class: 0

```



---



## Technologies



- Python 3.12

- scikit-learn

- NumPy

- SciPy

- Joblib

- Git

- GitHub



---



## Learning Goals



This project was created to practice the fundamentals of Machine Learning and build a strong foundation for more advanced projects.



Future projects will focus on topics such as:



- Data preprocessing

- Feature engineering

- Model comparison

- Hyperparameter tuning

- Data visualization

- Deep Learning

- Computer Vision



---



## Author



**Mohsen El-Sharnoby**



AI Student | Machine Learning Enthusiast

