import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Test accuracy: {accuracy:.2f}')

sample_index = 8
new_sample = X_test[sample_index].reshape(1, -1)
predicted_label, actual_label = model.predict(new_sample)[0], y_test[sample_index]

plt.imshow(new_sample.reshape(8, 8), cmap='gray_r')
plt.title(f'Predicted: {predicted_label}, Actual: {actual_label}')
plt.show()
