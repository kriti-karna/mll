import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('data.csv')
x=np.array(data)[:,:-1]
y=np.array(data)[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=109)


ml = GaussianNB()

ml.fit(x_train,y_train)
y_pred=ml.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
print('False Positives\n',FP)
print('False Negetives\n',FN)
print('True Positives\n',TP)
print('True Negetives\n',TN)
TPR = TP/(TP+FN)
print('Sensitivity \n',TPR)
TNR = TN/(TN+FP)
print('Specificity \n',TNR)
Precision = TP/(TP+FP)
print('Precision \n',Precision)
Recall = TP/(TP+FN)
print('Recall \n',Recall)
Acc = (TP+TN)/(TP+TN+FP+FN)
print('Áccuracy \n',Acc)
Fscore = 2*(Precision*Recall)/(Precision+Recall)
print('FScore \n',Fscore)
