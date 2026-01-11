import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']

features = ['age','trestbps','chol','thalach','oldpeak']

scaler = MinMaxScaler()

X_normalized = X.copy()
X_normalized[features] = scaler.fit_transform(X[features])
X_normalized.head()

scaler_z = StandardScaler()
X_standardized = X.copy()
X_standardized[features] = scaler_z.fit_transform(X[features])