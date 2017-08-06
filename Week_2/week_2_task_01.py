import numpy as np
import pandas
import sklearn
from sklearn.tree import DecisionTreeClassifier
import math


def get_optimal(data, result):
    kFold = sklearn.model_selection.KFold(shuffle=True, random_state=42, n_splits=5)
    best = -1
    ind = 0
    for i in range(1, 51):
        cl = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i)
        cvs = sklearn.model_selection.cross_val_score(cv=kFold, estimator=cl, X=data, y=result)
        ans = np.array(cvs).mean()
        if best < ans:
            best = ans
            ind = i
    return (best, ind)


lines = open("_805605c804bae8c5c24785f433b230ce_wine.data", "r").readlines()
all = [[float(y) for y in x.rstrip().split(",")] for x in lines]
data = [x[1:] for x in all]
result = [x[0] for x in all]

print(get_optimal(data, result))
data = sklearn.preprocessing.scale(X=data)
print(get_optimal(data, result))
