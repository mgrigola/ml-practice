import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
import cv2
from sklearn.externals import joblib


pickleFileClassifier = './data/save_classifier.pkl'

#iris = datasets.load_iris()
digits = datasets.load_digits()
N = len(digits.target)


#all the data?
#digits.data

#ground truth/true digit in each sample
#digits.target

# N_samples x (8,8)  : each image is 8x8 pixels with max intensity = 16
#digits.images

print('images', digits.images.shape)
print('data', digits.data.shape)
print('target', digits.target.shape)
print(digits.keys())
print(np.max(digits.images))
print(digits['target_names'])


svClassifier = svm.SVC(gamma=0.001, C=100.)

# do the learning
#svClassifier.fit(digits.data, digits.target)
svClassifier = joblib.load(pickleFileClassifier)

results = svClassifier.predict(digits.data)

#joblib.dump(svClassifier, pickleFileClassifier)

for idx in range(len(results)):
    if results[idx] != digits.target[idx]:
        print(idx, results[idx], digits.target[idx])

for idx,(pred, true) in enumerate(zip(results, digits.target)):
    if pred != true:
        print(idx, pred, true)

# cv2.namedWindow('img', 0)
# cv2.resizeWindow('img', 600, 600)
# for idx in range(len(results)):
#     if idx == 10:
#         break
#
#     cv2.imshow('img',digits.images[idx]/16.)
#     cv2.waitKey(400)
#     #print(idx, results[idx], digits.images[idx])
#
# cv2.destroyAllWindows()