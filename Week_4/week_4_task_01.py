import pandas
import numpy as np
import scipy
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

tfid = TfidfVectorizer(min_df=5)
enc_1 = DictVectorizer()
enc_2 = DictVectorizer()


def process_text(data, index, tv, test):
    text = data[index].replace('[^a-zA-Z0-9]', ' ', regex=True)
    if test:
        return tv.transform(text)
    else:
        return tv.fit_transform(text)


def process_word(data, index, enc, test):
    data[index].fillna('nan', inplace=True)
    words = data[[index]].to_dict('records')
    if test:
        return enc.transform(words)
    else:
        return enc.fit_transform(words)


def process_data(data, test=False):
    full_description = process_text(data, 'FullDescription', tfid, test)
    location_normalized = process_word(data, 'LocationNormalized', enc_1, test)
    contract_time = process_word(data, 'ContractTime', enc_2, test)
    return scipy.sparse.hstack([full_description, location_normalized, contract_time])


train_input = pandas.read_csv("_df0abf627c1cd98b7332b285875e7fe9_salary-train.csv")
test_input = pandas.read_csv("_d0f655638f1d87a0bdeb3bad26099ecd_salary-test-mini.csv")

tr_answer = train_input[train_input.columns[-1:]].get_values()

tr_data = process_data(train_input)
test_data = process_data(test_input, test=True)

ridge = Ridge(alpha=1, random_state=241)
ridge.fit(tr_data, tr_answer)

result = ridge.predict(test_data)
print(result)
