import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

for i, (true, pred) in enumerate(zip(y_test, predictions)):
    result = "Correct" if true == pred else "Wrong"
    print(f"Input: {X_test[i]}, Actual: {iris.target_names[true]}, Predicted: {iris.target_names[pred]} ({result})")

print("\nConfusion Matrix:\n", conf_matrix)
