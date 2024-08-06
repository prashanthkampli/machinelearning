import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('heart_disease_data.csv')

label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

X = data.drop(columns='heartDisease').values
y = data['heartDisease'].values

p_heart_disease = np.mean(y == 0)
p_no_heart_disease = np.mean(y == 1)

feature_probs = {}
for i, column in enumerate(data.columns[:-1]):
    feature_probs[column] = {}
    for value in np.unique(X[:, i]):
        p_value_given_hd = np.mean(X[y == 0, i] == value)
        p_value_given_no_hd = np.mean(X[y == 1, i] == value)
        feature_probs[column][value] = (p_value_given_hd, p_value_given_no_hd)

while True:
    inputs = [int(input(f'Enter {col} ({", ".join(map(str, np.unique(X[:, i])))}): ')) for i, col in enumerate(data.columns[:-1])]
    
    likelihood_hd = p_heart_disease
    likelihood_no_hd = p_no_heart_disease
    
    for i, col in enumerate(data.columns[:-1]):
        value = inputs[i]
        if value in feature_probs[col]:
            p_value_given_hd, p_value_given_no_hd = feature_probs[col][value]
            likelihood_hd *= p_value_given_hd
            likelihood_no_hd *= p_value_given_no_hd
    
    p_hd_given_inputs = likelihood_hd / (likelihood_hd + likelihood_no_hd)
    print(f"Probability of Heart Disease = {p_hd_given_inputs:.4f}")
    
    if int(input("Enter 0 to Continue, 1 to Exit: ")) == 1:
        break
