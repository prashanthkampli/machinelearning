from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

data = pd.read_csv('Housing.csv')
X = data[['area', 'bedrooms', 'bathrooms','storeys']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
