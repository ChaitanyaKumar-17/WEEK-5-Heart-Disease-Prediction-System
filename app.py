from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
  
# fetch the Heart Disease dataset
heart_disease = fetch_ucirepo(id=45) 
df = heart_disease.data.original
print(df.head())
print(df.info())

# Splitting dataDrame as features and target
X = heart_disease.data.features.copy()
print(X.head())
y = heart_disease.data.targets.copy()
print(y.head())

# Preprocessing
X['ca'] = X['ca'].fillna(X['ca'].mode()[0])
X['thal'] = X['thal'].fillna(X['thal'].mode()[0])

X['ca'] = X['ca'].astype('int64')
X['thal'] = X['thal'].astype('int64')

print(X.info())

# Split into training and testing sets to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)