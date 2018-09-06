import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("/Users/pascal_lin/Development/machine-learning/test/dataset/test.csv")
# print(dataset)

X = dataset.iloc[:, 1 : -3].values # all rows
print(X)

Y = dataset.iloc[ : , -1].values # pick the target column
print(Y)

# fixed missing number
imputer = preprocessing.Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 5:8]) # fixed column 6~9 number value
X[ : , 5:8] = imputer.transform(X[ : , 5:8]) # reload data

print(X)

# translate label to number

labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])

labelencoder_X = LabelEncoder()
X[ : , 1] = labelencoder_X.fit_transform(X[ : , 1])

labelencoder_X = LabelEncoder()
X[ : , 2] = labelencoder_X.fit_transform(X[ : , 2])

labelencoder_X = LabelEncoder()
X[ : , 3] = labelencoder_X.fit_transform(X[ : , 3])

labelencoder_X = LabelEncoder()
X[ : , 4] = labelencoder_X.fit_transform(X[ : , 4])

print(X)

# split dataset into test set and traning set

X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

print(X_test)
print(Y_test)

# feature standardization or Z-score normalization
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print(X_test)
print(X_train)