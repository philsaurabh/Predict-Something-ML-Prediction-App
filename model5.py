import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
data=pd.read_csv("glass.csv")

X=data.iloc[:,:-1]

y=data[["Type"]]


X=np.array(X)
y=np.array(y)

model5= LogisticRegression()
model5.fit(X,y.reshape(-1,))
joblib.dump(model5,"model5")