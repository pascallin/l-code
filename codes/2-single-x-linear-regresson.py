'''
y = b_0x + b_1
'''

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('/Users/pascal_lin/Development/machine-learning/test/dataset/studentscores.csv')
# print(dataset)

X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

# print(X)
# print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.25, random_state=0)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

regressor = LinearRegression()
regressor = regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_pred, color='blue')
plt.show()