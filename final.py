import numpy as np
import pandas as pd
file_path ="C://Users//naina//Downloads//healthcare.csv"
df = pd.read_csv(file_path)
#print(df.head())
df_clean = df.drop(columns=['ID','Heart Disease Ratio','Hypertension Ratio'])
print(df_clean.head())

df_clean['Bmi']=df_clean['Bmi'].str.replace(',','.').astype(float)
print(df_clean.head())
df_clean['HbA1c level'] = df_clean['HbA1c level'].str.replace(',', '.').astype(float)
print(df_clean.head())

#df_clean['Age'].fillna(df_clean['Age'].median(),inplace=True)
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
print(df_clean.head())
df_encoded = pd.get_dummies(df_clean, drop_first=True)
df_encoded.info,df_encoded.head()

# Save the cleaned dataset
output_path = "output.csv"
df_encoded.to_csv(output_path, index=False)

print(f"CSV file saved as: {output_path}")
output_path = "C://Users//naina//Downloads//output.csv"
df_encoded.to_csv(output_path, index=False)


