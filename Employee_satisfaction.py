import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from word2number import w2n
 

df = pd.read_csv('HR_exc.csv')


d = df[["satisfaction_level","average_montly_hours","promotion_last_5years","salary"]]
dum = pd.get_dummies(d.salary,dtype=int)
mer = pd.concat([d,dum],axis="columns")
fin = mer.drop(["salary"],axis="columns")
y = df.left
model = LogisticRegression(solver='lbfgs', max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(fin,y,test_size=0.02)
model.fit(X_train,y_train)
p = model.score(X_test,y_test)
print(p)

