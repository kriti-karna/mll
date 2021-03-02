from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import graphviz

data = pd.read_csv('zoo_data.csv')

train_features = np.array(data)[:80,:-1]
test_features = np.array(data)[80:,:-1]
train_targets = np.array(data)[:80,-1]
test_targets = np.array(data)[80:,-1]

tree1 = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)
export_graphviz(tree1, out_file="mytree.dot")
with open("mytree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

prediction = tree1.predict(test_features)
cm = confusion_matrix(test_targets, prediction)
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

print("The prediction accuracy is: ",tree1.score(prediction,test_targets)*100,"%")
