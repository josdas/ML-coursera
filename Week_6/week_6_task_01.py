import pandas
import numpy as np
import pylab as plt
import math
from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
from numpy import mean, median

N = 6
M = 7

def MSE(first, second):
    val = 0
    for i in range(len(first)):
        for k in range(3):
            val += (first[i][k] - second[i][k]) ** 2
    return val / len(first) / 3


def PSNR(first, second, maxi=1):
    mse = MSE(first, second)
    return 10 * math.log10(maxi ** 2 / mse)


def transform_img(data, group, function, k):
    clusters = [[] for i in range(k)]
    for i in range(len(group)):
        clusters[group[i]].append(i)
    new_img = np.zeros((len(data), 3))
    for i in range(k):
        color = [function(
            list(map(lambda x: data[x][j], clusters[i]))
        ) for j in range(3)]
        for j in clusters[i]:
            new_img[j] = color
    return new_img

image = img_as_float(imread('_3160f0832cf89866f4cc20e07ddf1a67_parrots.jpg'))

plt.ion()
plt.subplot(N, M, 1)
plt.imshow(image)

data = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
result = -np.inf
ind = -1
for i in range(1, 21):
    km = KMeans(n_clusters=i, init='k-means++', random_state=241)
    groups = km.fit_predict(data)

    mean_img = transform_img(data, groups, mean, i)
    median_img = transform_img(data, groups, median, i)

    mean_score = PSNR(mean_img, data)
    median_score = PSNR(median_img, data)

    print('i = {0}'.format(i))
    print('Mean score: {0}'.format(mean_score))
    print('Median score: {0}'.format(median_score))

    if result < mean_score or result < median_score:
        result = max(mean_score, median_score)
        ind = i

    plt.hold(True)

    plt.subplot(N, M, 2 * i)
    mean_data = mean_img.reshape((image.shape[0], image.shape[1], image.shape[2]))
    plt.imshow(mean_data)

    plt.subplot(N, M, 2 * i + 1)
    median_img = median_img.reshape((image.shape[0], image.shape[1], image.shape[2]))
    plt.imshow(median_img)

    plt.draw()
    plt.pause(0.1)

print(ind)
plt.show()
