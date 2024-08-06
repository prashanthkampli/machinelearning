from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data, target = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
model = svm.SVC(kernel='linear').fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Test accuracy: {accuracy:.2f}')

sample_index = 0
new_sample = X_test[sample_index].reshape(1, -1)
predicted_label, actual_label = model.predict(new_sample)[0], y_test[sample_index]

plt.barh(load_wine().feature_names, X_test[sample_index])
plt.title(f'Predicted: {load_wine().target_names[predicted_label]}, Actual: {load_wine().target_names[actual_label]}')
plt.xlabel('Scaled Feature Values')
plt.show()
