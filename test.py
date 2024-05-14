import pandas as pd
"""
Read the diabetes health indicator dataset 50:50 balanced.
Then:
- Remove rows containing null values
- Remove duplicate rows
- Check dataframe size just in case
"""
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
print(df.shape)
df = df.dropna()
print(df.shape)
df = df.drop_duplicates(ignore_index=True)
print(df.shape)

"""
Now prepare correlation matrix to show if features have any correlation with each other.
Prepare heatmap

"""

import matplotlib.pyplot as plt
import seaborn as sns
"""
corr_matrix = df.corr(method='pearson')  # 'pearson' is default

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r')
plt.show()
"""
"""
Now prepare correlation matrix between the target variable (Diabetes_binary) and each of the features
and present the heatmap

"""

corr_diabetes = df.copy()

corr_matrix = corr_diabetes.corr()

# Isolate the column corresponding to `Diabetes_binary`
corr_target = corr_matrix[['Diabetes_binary']].drop(labels=['Diabetes_binary'])

sns.heatmap(corr_target, annot=True, fmt='.3', cmap='RdBu_r')
plt.show()
