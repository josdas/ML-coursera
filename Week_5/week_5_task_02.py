import pandas
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def sigm(x):
    return 1 / (1 + math.exp(-x))


def log_loss_data(data, result):
    gr = []
    for val in data:
        tr_val = list(map(sigm, val))
        gr.append(log_loss(result, tr_val))
    return gr


def plot_draw(test_loss, train_loss):
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show(block=False)


def process(gbc, X, y):
    data = log_loss_data(gbc.staged_decision_function(X), y)
    min_ind = np.argmin(data)
    min_val = data[min_ind]
    return (min_val, min_ind, data)


input_data = pandas.read_csv("_75fb7a1b6f3431b6217cdbcba2fd30b9_gbm-data.csv")
X = input_data[input_data.columns[1:]].get_values()
y = input_data[input_data.columns[0]].get_values()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

best_value = [[] for i in range(5)]

for ind, i in enumerate([1, 0.5, 0.3, 0.2, 0.1]):
    gbc = GradientBoostingClassifier(n_estimators=250, verbose=False, random_state=241, learning_rate=i)
    gbc.fit(X_train, y_train)
    score_train = process(gbc, X_train, y_train)
    score_test = process(gbc, X_test, y_test)
    plot_draw(score_test[2], score_train[2])
    print("i = {0}".format(i))
    print("Train set. Min = {0}, argmin = {1}".format(score_train[0], score_train[1]))
    print("Test set. Min = {0}, argmin = {1}".format(score_test[0], score_test[1]))
    best_value[ind] = score_test[1]
    print(best_value[ind])

for i in best_value:
    if i > 0:
        rfc = RandomForestClassifier(random_state=241, n_estimators=i)
        rfc.fit(X_train, y_train)
        answer_train = rfc.predict_proba(X_test)
        lloss = log_loss(y_test, answer_train)
        print("Random forest with {0} tress has log_loss={1}".format(i, lloss))

plt.show(block=True)
