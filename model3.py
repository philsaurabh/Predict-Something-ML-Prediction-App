import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

data=pd.read_csv("iris.csv")

X=data.iloc[:,:-1]

mapping_target = {'Setosa':0, 'Versicolor':1, 'Virginica':2}
data = data.replace({'variety':mapping_target})
y=data[["variety"]]


X=np.array(X)
y=np.array(y)

model3= LogisticRegression()
model3.fit(X,y.reshape(-1,))
joblib.dump(model3,"model3")