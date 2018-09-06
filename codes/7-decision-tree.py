# -*- coding: utf-8 -*-  

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/Users/pascal_lin/Development/machine-learning/test/dataset/Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
X_set, y_set = X_test, y_pred
# meshgrid - 用向量生成矩阵
# arange - 生成数组，支持步长为小数
X1, X2 = np.meshgrid(
	np.arange(start=X_set[:,0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),
	np.arange(start=X_set[:,1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01)
)
# ravel - 拉平二维数组，二维变一维
# T - Transpose，转置矩阵
# contourf - 画三维等高线，第三个参数z为二维数组（表示平面点xi,yi映射的函数值）
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			alpha=0.75, cmap=ListedColormap(('red','green')))
# xlim - 设置x轴坐标
# ylim - 设置y轴坐标
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# 遍历目标值
for i,j in enumerate(np.unique(y_set)):
	# 生成坐标点
	plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1], c=ListedColormap(('red','green'))(i), label=j)
plt.title('Decision Tree Classification')
plt.xlabel('Age')
plt.ylabel('Estimate Salary')
plt.legend()
plt.show()