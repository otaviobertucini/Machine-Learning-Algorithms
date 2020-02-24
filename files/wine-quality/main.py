from utils.utils import open_cvs
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from pandas.plotting import scatter_matrix

data = open_cvs('/home/otavio/ml/datasets/wine-quality/winequality-red.csv')

data['fa_split'] = np.ceil(data['fixed acidity'])
data['ph_acidity'] = (data['pH']*data['volatile acidity'])/(data['alcohol'])
data['alcohol_acidity'] = (data['alcohol']/data['volatile acidity'])
data['']

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['fa_split']):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]

for sett in (train_set, test_set):
    sett.drop(['fa_split'], axis=1, inplace=True)

data = train_set.copy()
corr = data.corr()

print(corr['quality'].sort_values(ascending=False))

attributes = ['quality', 'alcohol', 'sulphates', 'citric acid', 'volatile acidity']
scatter_matrix(data[attributes])

# train_set.hist()
plt.savefig('hist.png')
