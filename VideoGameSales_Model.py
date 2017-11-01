#Video Game Sales Data model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('VideoGameSales.csv')
dataset = dataset.dropna()

y = dataset.iloc[:, 9].values
remCols = [0,5,6,7,8,9,12,13,14,15]
dataset.drop(dataset.columns[remCols],axis=1,inplace=True)
#dataset = pd.get_dummies(dataset, drop_first = True)

X = dataset.iloc[:, :].values

y = (y > 1)

'''
#Uncomment to handle categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [0, 2, 3, 6])
X = onehotencoder.fit_transform(X).toarray()

#Remove one of the rows the avoid the Dummy Variable trap
X = X[:, 1:]
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras 
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(activation = 'relu', input_dim = 6, units=1000, kernel_initializer="uniform"))
classifier.add(Dense(activation = 'relu', units=1000, kernel_initializer="uniform"))
classifier.add(Dense(activation = 'sigmoid', units=1000, kernel_initializer="uniform"))
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer="uniform")) 
#metric vs loss: loss function is used in training, metric isn't
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Predicting a single new observation
'''
new_prediction = classifier.predict(sc_X.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > .5)
'''

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

weights = classifier.model.get_weights()
sum0 = np.sum(weights[0][3])