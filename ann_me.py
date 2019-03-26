# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 23:20:49 2018

@author: Prachi
"""

# Part 1 - DATA PREPROCESSING

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder_X1 = LabelEncoder()
label_encoder_X2 = LabelEncoder()
X[:,1] = label_encoder_X1.fit_transform(X[:,1]) 
X[:,2] = label_encoder_X2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#Apply Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()
classifier.add(Dense(input_dim = 11,activation = "relu", kernel_initializer="uniform", units=6))
classifier.add(Dense(activation = "relu", kernel_initializer="uniform", units=6))
classifier.add(Dense(activation = "sigmoid", kernel_initializer="uniform", units=1))

#Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting ANN to the Training Set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#Predicting the Test Result
pred = classifier.predict(X_test)
pred = pred > 0.5


#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
