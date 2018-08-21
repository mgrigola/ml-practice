import numpy as np
from sklearn import random_projection
from sklearn import datasets
from sklearn.svm import SVC

from matplotlib import pyplot


irisData = datasets.load_iris()
clf = SVC()

print(irisData.keys())


# rng = np.random.RandomState(None)
# x = np.array(rng.rand(10, 2000))

# projTransformer = random_projection.GaussianRandomProjection()
#
# xTransf = projTransformer.fit_transform(x)
#
# print('x type:', x.dtype)
# print('\nx', x)
# print('\nxtransf type:', xTransf.dtype)
# print('\nxTransf',xTransf)


# cnts, bins = np.histogram(x, 100)
#
# pyplot.plot( (bins[:-1]+bins[1:])/2. , cnts)
# pyplot.show()
# #
# cnts, bins = np.histogram(xTransf, 100)
#
# plt.plot( (bins[:-1]+bins[1:])/2. , cnts)
# plt.show()