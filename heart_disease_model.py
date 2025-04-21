import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier
heart_disease = pd.read_csv("heartdisease.csv")
#print(heart_disease.head())
#print(heart_disease.shape)
#print(heart_disease.describe())
#print(heart_disease['target'].value_counts())
X=heart_disease.drop(columns='target',axis=1)
Y=heart_disease['target']
#print(x,y)
scalar = StandardScaler()
standard_data=scalar.fit_transform(X)
#print(standard_data)
X=standard_data
#print(x,y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=2)
#print(x.shape,x_test.shape,x_train.shape)

knn=KNeighborsClassifier(n_neighbors=25,metric='minkowski')
knn.fit(X_train,Y_train)

Y_pred=knn.predict(X_test)

#print("train accuracy :",train_data_accuracy)

#print("test accuracy :",test_data_accuracy)

pickle.dump(scalar,open('heart_disease_scalar.pkl','wb'))
pickle.dump(knn,open('heart_disease_model.pkl','wb'))





