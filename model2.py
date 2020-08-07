import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("heart.csv")
data["trestbps"]=np.log(data["trestbps"])

data=data.drop(["fbs"],axis=1)
data=data.drop(["ca"],axis=1)
data["chol"]=np.log(data["chol"])
target=data["target"]

np.random.shuffle(data.values)
data=data.drop(["target"],axis=1)
sc= StandardScaler()
data=sc.fit_transform(data)

model2=SGDClassifier(loss="hinge", penalty="l2", max_iter=7)
model2.fit(data,target)
cv_results = cross_validate(model2, data,target, cv=10)
joblib.dump(model2,"model2")
