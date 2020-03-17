from sklearn import datasets
import matplotlib
import matplotlib.pyplot as ptl
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from utils.utils import open_cvs

data = open_cvs('/home/otavio/ml/datasets/fake-job/fake_job_postings.csv')

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(train_set.drop('fraudulent', 1), train_set['fraudulent'])
