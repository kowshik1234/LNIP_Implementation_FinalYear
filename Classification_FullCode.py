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
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

x = pd.read_csv(r'F:\4th_Year_project_files\Codes\All_dataset.csv')

training_set,test_set = train_test_split(x,test_size=0.2,random_state=0)
X_train = training_set.iloc[:,0:59].values
Y_train = training_set.iloc[:,59].values
X_test = test_set.iloc[:,0:59].values
Y_test = test_set.iloc[:,59].values

#####################Linear SVM Classifier###############
SVM = svm.LinearSVC()
SVM.fit(X_train,Y_train)
SVM_pred = SVM.predict(X_test)
cm = confusion_matrix(Y_test,SVM_pred)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Linear SVM For The Given Dataset: ", accuracy)

#####################SVM with Radial Basis Function Kernel##############
classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train,Y_train)
rbf_predict = classifier.predict(X_test)
cm = confusion_matrix(Y_test,rbf_predict)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Radial Basis Function For The Given Dataset : ", accuracy)

##################### Random Forest Classifier#################
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)


#####################Making Predictions##################
rfc_predict = rfc.predict(X_test)
cm = confusion_matrix(Y_test,rfc_predict)
accuracy = float(cm.diagonal().sum())/len(Y_test)
print("\nAccuracy Of Random Forest For The Given Dataset :", accuracy)



##rfc = RandomForestClassifier()
##rfc.fit(X,y)
##rfc_predict = rfc.predict(test)
##print(rfc_predict)
##rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
##
##print(rfc_cv_score)

##a = np.array(x)
##y = a[:,59]
##
##print(x)
