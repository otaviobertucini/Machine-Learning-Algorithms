import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from utils.utils import open_cvs
from sklearn.neighbors import KNeighborsClassifier

train_set = open_cvs('/home/otavio/ml/datasets/numbers/mnist_train.csv')
test_set = open_cvs('/home/otavio/ml/datasets/numbers/mnist_test.csv')

train_data = train_set.drop('label', axis=1)
train_labels = train_set['label']

test_data = test_set.drop('label', axis=1)
test_labels = test_set['label']

knn = KNeighborsClassifier()
knn.fit(train_data[:30000], train_labels[:30000])

scores = cross_val_score(knn, train_data, train_labels, scoring="accuracy", cv=3)
print(scores)
