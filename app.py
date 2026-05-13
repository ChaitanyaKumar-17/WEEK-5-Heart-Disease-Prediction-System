import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  
# fetch the Heart Disease dataset
heart_disease = fetch_ucirepo(id=45) 
df = heart_disease.data.original
print(df.head())
print(df.info())

# Splitting dataDrame as features and target
X = heart_disease.data.features.copy()
print(X.head())
y = heart_disease.data.targets.copy()
y = (y > 0).astype(int)
print(y.head())

# Preprocessing
X['ca'] = X['ca'].fillna(X['ca'].mode()[0])
X['thal'] = X['thal'].fillna(X['thal'].mode()[0])

X['ca'] = X['ca'].astype('int64')
X['thal'] = X['thal'].astype('int64')

print(X.info())

# Split into training and testing sets to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column transformation
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
passthrough_cols = ['sex', 'fbs', 'exang', 'ca']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_cols),
        # drop='first' prevents multicollinearity (dummy variable trap)
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols) 
    ],
    remainder='passthrough'
)

# Data transformation
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Training and testing the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # max_iter is increased to 1000 to ensure the math algorithm has time to converge
    ('classifier', LogisticRegression(max_iter=1000, random_state=42)) 
])

pipeline.fit(X_train, np.ravel(y_train))

y_pred = pipeline.predict(X_test)

# Model Evaluation
print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
