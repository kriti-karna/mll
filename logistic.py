import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('StudentUniversitylogistic.csv')
x=np.array(data)[:,:-1]
y=np.array(data)[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=109)


classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm, "\n")
fp=cm.sum(axis=0)-np.diag(cm)
fn=cm.sum(axis=1)-np.diag(cm)
tp=np.diag(cm)
tn=cm.sum()-(fp+fn+tp)
print("false positives:",fp)
print("false negatives:",fn)
print("true positives:",tp)
print("true negatives:",tn)
tnr = tn/(tn+fp)
print("tnr:",tnr)
tpr = tp/(tp+fn)
print("tpr:",tpr)
acc = (tp+tn)/(tp+tn+fp+fn)
print("acc:",acc)
recall = (tp/(tp+fp))
print("recall:",recall)
precision = (tp/(tp+fn))
print("precision:",precision)
Fscore = 2*(precision*recall)/(precision+recall)
print("Fscore:",Fscore)
