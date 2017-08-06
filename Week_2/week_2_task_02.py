import numpy as np
import pandas
import sklearn
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
import math

boston = sklearn.datasets.load_boston()
boston.data = sklearn.preprocessing.scale(boston.data)

kFold = sklearn.model_selection.KFold(shuffle=True, random_state=42, n_splits=5)
result = np.inf
ind = -1
for i in np.linspace(1, 10, num=200):
    knr = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', p=i)
    cvs = sklearn.model_selection.cross_val_score(cv=kFold, estimator=knr, X=boston.data, y=boston.target,
                                                  scoring='neg_mean_squared_error')
    ans = np.array(cvs).mean()
    if result > ans:
        result = ans
        ind = i
print(ind)
