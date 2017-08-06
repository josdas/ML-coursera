import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
import math

data = pandas.read_csv('_ea07570741a3ec966e284208f588e50e_titanic.csv', index_col='PassengerId')

survived_data = data["Survived"].tolist()
pclass_data = data["Pclass"].tolist()
fare_data = data["Fare"].tolist()
age_data = data["Age"].tolist()
sex_data = list(map(lambda x: True if x == "male" else False, data["Sex"].tolist()))

c_data = []
r_data = []

for i in range(len(age_data)):
    if not np.isnan(age_data[i]) and not np.isnan(fare_data[i]) and not np.isnan(pclass_data[i]):
        c_data.append([pclass_data[i], fare_data[i], age_data[i], sex_data[i]])
        r_data.append(survived_data[i])

clf = DecisionTreeClassifier(random_state=241)
clf.fit(c_data, r_data)

print(clf.feature_importances_)