'''
classification problems
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('/Users/pascal_lin/Development/machine-learning/test/dataset/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# visualization

plt.title('Training set')
plt.xlabel('age')
plt.ylabel('estimated salary')

for index, value in enumerate(y_train):
	color = 'green' if value > 0 else 'red'
	plt.scatter(X_train[index][0], X_train[index][1], color=color)

plt.show()

plt.title('Test set')
plt.xlabel('age')
plt.ylabel('estimated salary')

for index, value in enumerate(y_pred):
	color = 'green' if value > 0 else 'red'
	plt.scatter(X_test[index][0], X_test[index][1], color=color)

plt.show()

# Making the Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)