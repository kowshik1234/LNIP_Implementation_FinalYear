import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")

x = pd.read_csv(r'F:\4th_Year_project_files\Dataset_DDSM_database\LNIP_all.csv')

training_set,test_set = train_test_split(x,test_size=0.2,random_state=0)
X_train = training_set.iloc[:,0:256].values
Y_train = training_set.iloc[:,256].values
X_test = test_set.iloc[:,0:256].values
Y_test = test_set.iloc[:,256].values

classifier = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=10, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

classifier.fit(X_train,Y_train)
SVM_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test,SVM_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Linear SVM For The Given Dataset: ", accuracy)

#########################Linear SVM Classifier###############
SVM = svm.LinearSVC()
SVM.fit(X_train,Y_train)
SVM_pred = SVM.predict(X_test)
cm = confusion_matrix(Y_test,SVM_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Linear SVM For The Given Dataset: ", accuracy)

#########################SVM with Radial Basis Function Kernel##############
classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train,Y_train)
rbf_predict = classifier.predict(X_test)
cm = confusion_matrix(Y_test,rbf_predict)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Radial Basis Function For The Given Dataset : ", accuracy)

######################### Random Forest Classifier#################
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
##
##
###############################Kitchen Sink################
rbf_feature = RBFSampler(gamma=0.001,n_components=59 ,random_state=0)
X_features = rbf_feature.fit_transform(X_train)
clf = SGDClassifier(max_iter=20,tol=1)
clf.fit(X_features, Y_train)
res = clf.predict(X_test)
cm = confusion_matrix(Y_test,res)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Random Forest For The Given Dataset :", accuracy)
print(res)
##
#########################Making Predictions##################
rfc_predict = rfc.predict(X_test)
cm = confusion_matrix(Y_test,rfc_predict)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Random Forest For The Given Dataset :", accuracy)
######################### XG Boost############################
model = XGBClassifier()
model.fit(X_train, Y_train)
# make predictions for test data
y_pred = model.predict(X_test)
##cm = confusion_matrix(Y_test,y_pred)
##accuracy = float(cm.diagonal().sum())/len(Y_test)
##print("Accuracy is ->",accuracy)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


