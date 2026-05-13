from ucimlrepo import fetch_ucirepo 
  
# fetch the Heart Disease dataset
heart_disease = fetch_ucirepo(id=45) 
df = heart_disease.data.original
print(df.head())
# Extract data as pandas dataframes 
# X = heart_disease.data.features 
# y = heart_disease.data.targets 
