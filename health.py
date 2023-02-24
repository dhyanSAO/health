import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
health_data = pd.read_csv('/content/archive.zip')

# print first 5 rows of the dataset
health_data.head()

# print last 5 rows of the dataset
health_data.tail()

# number of rows and columns in the dataset
health_data.shape

# getting some info about the data
health_data.info()

# checking for missing values
health_data.isnull().sum()

# statistical measures about the data
health_data.describe()

# checking the distribution of Target Variable
health_data['RiskLevel'].value_counts()

X = health_data.drop(columns='RiskLevel', axis=1)
Y = health_data['RiskLevel']

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

input_data = (25,130,80,	15.0,	98.0	,86	)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
