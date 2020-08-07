import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,SGDClassifier
import joblib
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("diabetes.csv")

model=SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

X=data.iloc[:,:8]

y=data[["Outcome"]]

X=np.array(X)
y=np.array(y)

model.fit(X,y.reshape(-1,))
joblib.dump(model,"model")