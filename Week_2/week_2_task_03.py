import numpy as np
import pandas
import sklearn
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from sklearn.metrics import accuracy_score


def get_data(data):
    return [[x[1:] for x in data.get_values()], [x[0] for x in data.get_values()]]


def score(test, train):
    clf = Perceptron(random_state=241)
    clf.fit(train[0], train[1])
    result = clf.predict(test[0])
    return accuracy_score(test[1], result)


train = get_data(read_csv("_3abd237d917280ba0d83bfe6bd49776f_perceptron-train.csv", header=None))
test = get_data(read_csv("_3abd237d917280ba0d83bfe6bd49776f_perceptron-test.csv", header=None))

first_result = score(test, train)

scaler = StandardScaler()
scaler.fit_transform(np.array(train[0]))
train[0] = scaler.transform(train[0])
test[0] = scaler.transform(test[0])

print((score(test, train) - first_result) * 100)
