# Heart Disease Prediction with Machine Learning

This notebook demonstrates a machine learning workflow to predict heart disease based on various health parameters. The project uses a Logistic Regression model to classify individuals as 'Healthy' or having 'Heart Disease'.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Workflow](#workflow)
- [Model Performance](#model-performance)
- [How to Use](#how-to-use)

## Project Overview

This project aims to build and evaluate a predictive model for heart disease. By analyzing medical data, the model helps in identifying individuals at risk, which can assist healthcare professionals in early diagnosis and intervention.

## Dataset

The dataset used for this project is `heart_disease_dataset.csv`. It contains the following features:

- `age`: Age in years
- `sex`: Sex (1 = male; 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholestoral (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes; 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: The slope of the peak exercise ST segment (0-2)
- `ca`: Number of major vessels (0-3) colored by flourosopy
- `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
- `target`: Diagnosis of heart disease (1 = heart disease; 0 = no heart disease)

## Libraries Used

The following Python libraries are used in this notebook:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib.pyplot`: For data visualization.
- `seaborn`: For enhanced data visualization.
- `sklearn.model_selection`: For splitting data and cross-validation (`train_test_split`, `cross_val_score`).
- `sklearn.linear_model`: For the Logistic Regression model.
- `sklearn.metrics`: For model evaluation (`accuracy_score`, `confusion_matrix`, `classification_report`).
- `sklearn.preprocessing`: For data scaling (`StandardScaler`).
- `joblib`: For saving and loading the model and scaler.

## Workflow

The notebook follows a standard machine learning pipeline:

1.  **Data Loading**: The `heart_disease_dataset.csv` is loaded into a pandas DataFrame.
2.  **Exploratory Data Analysis (EDA)**:
    *   Initial inspection of data using `df.head()`, `df.shape`, `df.info()`, `df.describe()`.
    *   Checking for missing values (`df.isnull().sum()`).
    *   Analyzing the balance of the target variable (`df['target'].value_counts()`).
    *   Visualizing feature distributions (`age`, `chol`, `thalach`) relative to the target variable using histograms.
    *   Generating a correlation heatmap to understand relationships between features.
3.  **Data Preprocessing**:
    *   Separating features (`X`) and target (`y`).
    *   Splitting the data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using `train_test_split` with stratification.
    *   Scaling numerical features using `StandardScaler` to normalize the data.
    *   Saving the trained `StandardScaler` object (`heart_scaler.pkl`).
4.  **Model Training**: A Logistic Regression model is trained on the scaled training data.
    *   Cross-validation is performed to assess model robustness.
5.  **Model Evaluation**: The trained model is evaluated on both training and testing datasets.
    *   Accuracy scores are calculated.
    *   A classification report (precision, recall, f1-score) is generated for the test set.
    *   A confusion matrix is visualized to show true positives, true negatives, false positives, and false negatives.
6.  **Model Saving**: The trained Logistic Regression model and feature names are saved using `joblib` (`heart_disease_model.pkl`).
7.  **Prediction on New Data**: The notebook includes a section to take user input for new patient data and predict their heart disease risk.

## Model Performance

- **Cross-validation Accuracy**: 0.8309 (+/- 0.0409)
- **Train Accuracy**: 0.8430
- **Test Accuracy**: 0.8033

The classification report for the test set is as follows:

```
              precision    recall  f1-score   support

     Healthy       0.86      0.68      0.76        28
     Disease       0.77      0.91      0.83        33

    accuracy                           0.80        61
   macro avg       0.82      0.79      0.80        61
weighted avg       0.81      0.80      0.80        61
```

## How to Use

1.  **Clone the repository/Download the notebook**.
2.  **Ensure you have the dataset**: Place `heart_disease_dataset.csv` in the same directory as the notebook or update the path in the code.
3.  **Install required libraries**: 
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```
4.  **Run the notebook**: Execute all cells in the notebook sequentially.
5.  **Interactive Prediction**: Use the last code cell to input new patient data and get a heart disease risk prediction.