import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier

lung_cancer = pd.read_csv("lung_cancer.csv") 
# print(heart_stroke.head()) 
# print(heart_stroke.shape) 
# print(heart_stroke.describe())   
# print(heart_stroke['Outcome'].value_counts())   
X=lung_cancer.drop(columns=['GENDER1','LUNG_CANCER1','LUNG_CANCER'],axis=1) 
Y=lung_cancer['LUNG_CANCER']
# print(X,Y)

scalar = StandardScaler()
standard_data=scalar.fit_transform(X)

# print(standard_data)

X=standard_data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=2)

# print(X.shape,X_test.shape,X_train.shape)

# classifier=svm.SVC(kernel="linear")
# classifier.fit(X_train,Y_train)

# X_train_prediction=classifier.predict(X_train)
# train_accuracy=accuracy_score(X_train_prediction,Y_train)
# print("accuracy is",train_accuracy) 

# X_test_prediction=classifier.predict(X_test)
# test_accuracy=accuracy_score(X_test_prediction,Y_test)
# print("accuracy is",test_accuracy)

knn=KNeighborsClassifier(n_neighbors=25,metric='minkowski')
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)

pickle.dump(scalar,open('lung_cancer_scalar.pkl','wb'))
pickle.dump(knn,open('lung_cancer_model.pkl','wb'))

