import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
%matplotlib inline
import sklearn as sns

df=pd.read_csv("E:/diabetes.csv")

df.plot(kind='density',subplots=True,layout=(3,3),sharex=False)

df.isnull().sum()

df.describe()

df.Outcome.value_counts()

x=df.drop('Outcome',axis=1)

from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()

df['Outcome']=Le.fit_transform(df['Outcome'])

y=df.Outcome

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler

x_scaler=scaler.fit_transform(x)

x_scaler

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaler,y,stratify=y,random_state=10)

x_test.shape

y_train.value_counts()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

score = cross_val_score(DecisionTreeClassifier(),x,y,cv=5)
score

score.mean()

from sklearn.ensemble import BaggingClassifier


bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                           n_estimators=100,
                           max_samples=0.8,
                           oob_score=True,
                           random_state=0)
bag_model.fit(x_train, y_train)
bag_model.oob_score_

bag_model.score(x_test,y_test)

x_test

bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                           n_estimators=100,
                           max_samples=0.8,
                           oob_score=True,
                           random_state=0)
scores=cross_val_score(bag_model,x,y,cv=5)
scores

scores.mean()

from sklearn.ensemble import  RandomForestClassifier
scores = cross_val_score( RandomForestClassifier(n_estimators=50),x,y,cv=5)
scores

scores.mean()

df
