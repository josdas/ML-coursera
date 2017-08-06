import pandas
import numpy as np
import pylab as plt
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

input_data = pandas.read_csv("_8d955d45315ff739d75fd4de3c97acf9_abalone.csv")
input_data['Sex'] = input_data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

features = input_data[input_data.columns[:-1]].get_values()
answer = input_data[input_data.columns[-1]].get_values()

ind = -1
kf = KFold(random_state=1, shuffle=True, n_folds=5, n=len(features))
for i in range(1, 51):
    RFR = RandomForestRegressor(random_state=1, n_estimators=i)
    val = np.mean(cross_val_score(RFR, features, answer, 'r2', kf))

    print("i = {0}\nScore = {1}".format(i, val))
    plt.scatter(i, val)

    if val > 0.52 and ind == -1:
        ind = i
print(ind)
plt.show()
