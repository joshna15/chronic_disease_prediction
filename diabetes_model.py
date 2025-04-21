import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier
diabetes_dataset=pd.read_csv("diabetes.csv")
# print(diabetes_data.head())
# print(diabetes_data.describe())
# print(diabetes_data.groupby("Outcome").mean())
X=diabetes_dataset.drop(columns="Outcome",axis=1)
Y=diabetes_dataset['Outcome']
# print(X)
# print(Y)
scalar=StandardScaler()

standardised_data=scalar.fit_transform(X)
# print(standardised_data)

X=standardised_data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# print(X.shape,X_test.shape,X_train.shape)
# classifier=svm.SVC(kernel="linear")
# classifier.fit(X_train,Y_train)

# X_train_prediction=classifier.predict(X_train)
# train_accuracy=accuracy_score(X_train_prediction,Y_train)
# print("accuracy is",train_accuracy)

knn=KNeighborsClassifier(n_neighbors=25,metric='minkowski')
knn.fit(X_train,Y_train)

Y_pred=knn.predict(X_test)
# print(Y_pred)

# print(accuracy_score(Y_test,Y_pred))



# X_test_prediction=classifier.predict(X_test)
# test_accuracy=accuracy_score(X_test_prediction,Y_test)
# print("accuracy is",test_accuracy)

# input_data=(4,110,92,0,0,37.6,30)
# input_data_as_numpy_array=np.asarray(input_data)
# input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
# std_data=scalar.transform(input_data_reshaped)
# print(std_data)
# prediction=classifier.predict(std_data)
# print(prediction)

pickle.dump(scalar,open('diabetes_scalar.pkl','wb'))
pickle.dump(knn,open('diabetes_model.pkl','wb'))