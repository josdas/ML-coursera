from sklearn.svm import SVC
import pandas
import numpy as np

data = pandas.read_csv("_f6284c13db83a3074c2b987f714f24f5_svm-data.csv", header=None)
points = data[data.columns[1:]]
result = data[data.columns[0]]
svc = SVC(C=100000, random_state=241, kernel='linear')
svc.fit(X=points, y=result)
print(svc.support_)
