# Heart Disease Prediction System Using Logistic Regression

A robust Python machine learning project designed to ingest clinical patient data, preprocess mixed data types (continuous and categorical), and predict the presence of heart disease using a Logistic Regression model. This project emphasizes industry best practices in building scalable, leak-free machine learning pipelines.

## 💡 Overview

Predictive analytics in healthcare requires strict data hygiene and mathematically sound preprocessing. This project automates the end-to-end workflow of a supervised learning classifier. It fetches the standard UCI Heart Disease dataset, handles missing values, dynamically applies distinct scaling and encoding rules to different feature types, and outputs reliable diagnostic predictions. By optimizing the target variable for binary classification, the model achieves high accuracy while avoiding common data sparsity errors.

## ✨ Features

* **Automated Data Acquisition:** Directly fetches the verified Heart Disease dataset from the UCI Machine Learning Repository (`ucimlrepo`), ensuring data authenticity.
* **Target Optimization:** Converts the original 5-class severity scale into a clean binary classification problem (Healthy vs. Disease), drastically improving model accuracy and preventing zero-division metric errors.
* **Robust Preprocessing Engine:** Utilizes `scikit-learn`'s `ColumnTransformer` to independently route continuous features through a `StandardScaler` and categorical features through a `OneHotEncoder`, all in parallel.
* **Leak-Free Architecture:** Wraps the preprocessor and the `LogisticRegression` algorithm inside a unified `Pipeline`. This guarantees that no information from the testing set "leaks" into the training set during mathematical scaling.
* **Objective Evaluation:** Generates a comprehensive clinical assessment of the model, outputting the exact Accuracy Score, a standard Confusion Matrix, and a detailed Classification Report (Precision, Recall, F1-Score).

## 🛠️ Prerequisites

* Python 3.8 or higher
* A standard Python IDE (VS Code, PyCharm) or Jupyter Notebook
* Core Scientific Libraries: `pandas`, `numpy`, `scikit-learn`, `ucimlrepo`

## 🚀 Usage

1. Clone this repository to your local machine.
2. Open your terminal or command prompt and install the necessary dependencies:
   ```bash
   pip install pandas numpy scikit-learn ucimlrepo
   ```
3. Launch your Python environment or execute the script directly:
   ```bash
   python app.py
   ```

## 📊 Expected Output

Upon successful execution, the script will process the raw clinical data in memory and output fact-grounded terminal metrics, including:

1. **Dataset Diagnostics:** Terminal output showing the `head()` and `info()` of the processed DataFrame, verifying that NaNs have been imputed and data types cast correctly.

2. **Model Evaluation Metrics:** Terminal output stating the exact calculated Accuracy Score (typically stabilizing around 80-85%).

3. **Confusion Matrix & Classification Report:** A 2x2 grid displaying True Positives/Negatives against False Positives/Negatives, alongside Recall and Precision metrics to evaluate the model's clinical viability.

## 🧩 How It Works (Under the Hood)

This script serves as a practical application of standard machine learning pipelines:

1. **Data Ingestion & Cleaning:** The script loads dataset ID 45 from the UCI repository. It locates missing values in the `ca` and `thal` columns and statistically imputes them using the mode, ensuring the dataset is dense and calculable.

2. **Train/Test Splitting:** The algorithm splits the data into training (80%) and testing (20%) subsets before any scaling occurs to preserve the integrity of the test environment.

3. **Column Transformation:** Continuous clinical features (like cholesterol and heart rate) are scaled to have a mean of 0 and a standard deviation of 1. Categorical features (like chest pain type) are transformed into binary matrices, strategically dropping the first column to avoid the dummy-variable trap (multicollinearity).

4. **Pipeline Execution & Convergence:** The transformed data flows directly into a Logistic Regression classifier. The model's `max_iter` parameter is increased to 1000, ensuring the gradient descent algorithm has sufficient time to mathematically converge on the optimal weights.