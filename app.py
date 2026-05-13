from ucimlrepo import fetch_ucirepo 
  
# fetch the Heart Disease dataset
heart_disease = fetch_ucirepo(id=45) 
df = heart_disease.data.original
print(df.head())
print(df.info())

# Splitting dataDrame as features and target
X = heart_disease.data.features
print(X.head())
y = heart_disease.data.targets
print(y.head())
