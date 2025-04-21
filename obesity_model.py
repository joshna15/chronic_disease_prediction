import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier
obesity = pd.read_csv("obesity.csv") 
# print(obesity.head()) 
# print(obesity.shape) 
# print(obesity.describe())   
# print(obesity['ObesityCategory'].value_counts())   
X=obesity.drop(columns=['ObesityCategory1','Gender1','ObesityCategory'],axis=1) 
Y=obesity['ObesityCategory']

# print(X,Y)  

scalar = StandardScaler()
standard_data=scalar.fit_transform(X)

# print(standard_data)

X=standard_data

# print(X,Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=2)

# print(X.shape,X_test.shape,X_train.shape)

knn=KNeighborsClassifier(n_neighbors=25,metric='minkowski')
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)

# classifier=svm.SVC(kernel="linear")
# classifier.fit(X_train,Y_train)

# X_train_prediction=classifier.predict(X_train)
# train_accuracy=accuracy_score(X_train_prediction,Y_train)
# print("accuracy is",train_accuracy) 

# X_test_prediction=classifier.predict(X_test)
# test_accuracy=accuracy_score(X_test_prediction,Y_test)
# print("accuracy is",test_accuracy)

pickle.dump(scalar,open('obesity_scalar.pkl','wb'))
pickle.dump(knn,open('obesity_model.pkl','wb'))

# input_data=(56,1,173.5752624,71.98205082,23.89178262)
# input_data_as_numpy_array=np.asarray(input_data)
# input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
# std_data=scalar.transform(input_data_reshaped)
# print(std_data)
# prediction=classifier.predict(std_data)
# print(prediction)

