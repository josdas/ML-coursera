from sklearn.model_selection import GridSearchCV
import pandas
import sklearn.metrics as metrics
import numpy as np

data = pandas.read_csv('_8b9c6d9ae39e206610c6fd96894615a5_classification.csv')

result = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
for i in range(len(data.index)):
    true = data['true'][i]
    pred = data['pred'][i]
    result['tp'] += 1 if true == 1 and pred == 1 else 0
    result['fp'] += 1 if true == 0 and pred == 1 else 0
    result['fn'] += 1 if true == 1 and pred == 0 else 0
    result['tn'] += 1 if true == 0 and pred == 0 else 0
print('{tp} {fp} {fn} {tn}'.format(**result))

print(metrics.accuracy_score(data['true'], data['pred']),
      metrics.precision_score(data['true'], data['pred']),
      metrics.recall_score(data['true'], data['pred']),
      metrics.f1_score(data['true'], data['pred']))

scores = pandas.read_csv('_eee1b9e8188f61bc35d954fbeb94e325_scores.csv')

max_val = -1
max_name = ''
for x in set(scores.columns) - {'true'}:
    auc = metrics.roc_auc_score(scores['true'], scores[x])
    if max_val < auc:
        max_val = auc
        max_name = x
print(max_name)

max_val = -1
max_name = ''
for x in set(scores.columns) - {'true'}:
    curve = metrics.precision_recall_curve(scores['true'], scores[x])
    for pres, rec, thr in zip(*curve):
        if rec > 0.7:
            if max_val < pres:
                max_val = pres
                max_name = x
print(max_name)