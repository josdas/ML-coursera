from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold


def find_best(data_, answer_):
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    kFold = KFold(shuffle=True, random_state=241, n_splits=5)
    svc_grid = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(svc_grid, grid, scoring='accuracy', cv=kFold)
    gs.fit(data_, answer_)
    return gs.best_params_['C']


newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
tfv = TfidfVectorizer()
answer = newsgroups.target
data = tfv.fit_transform(newsgroups.data, answer)
feature_mapping = tfv.get_feature_names()

c_optimal = 1
svc = SVC(kernel='linear', C=c_optimal, random_state=241)
svc.fit(data, answer)
svc_result = svc.coef_.toarray()[0]

result = list(range(svc_result.size))
result.sort(reverse=True, key=lambda el: abs(svc_result[el]))

final_answer = sorted(map(lambda el: feature_mapping[el], result[:10]))
for x in final_answer:
    print(x, end=',')
