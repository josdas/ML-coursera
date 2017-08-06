import pandas
import numpy as np
from sklearn.decomposition import PCA

input_prices = pandas.read_csv("_82b4a9f66c689b3d40dd25ebd761b07f_close_prices.csv")
input_djia = pandas.read_csv("_82b4a9f66c689b3d40dd25ebd761b07f_djia_index.csv")
prices = input_prices[input_prices.columns[1:]].get_values()
djia = input_djia[input_djia.columns[1]].get_values()

pca = PCA(n_components=10)
pca.fit(prices)

# first task
sum_disp = 0
i = 0
while sum_disp < 0.9:
    sum_disp += pca.explained_variance_ratio_[i]
    i += 1
print(i)

# second task
tr_prices = pca.transform(prices)
first_column = tr_prices[:, 0]
print(np.corrcoef(djia, first_column)[0][1])

# third task
first_component = pca.components_[0]
index_max = np.argmax(first_component)
print(input_prices.columns[index_max + 1])
