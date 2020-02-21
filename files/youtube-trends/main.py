from utils.utils import open_cvs
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
plt.interactive(True)

data = open_cvs('/home/otavio/ml/datasets/youtube-trends/usvideos.csv')

# print(data.head())
# print(data.info())
# print(data['category_id'].value_counts())
# print(data.describe())

# data['likes'].where(data['likes'] < 100000, 100000, inplace=True)
# data['likes'].where(data['likes'] > 100000, 100000, inplace=True)
# data['dislikes'].where(data['dislikes'] < 2500, 2500, inplace=True)
# data['comment_count'].where(data['comment_count'] < 10000, 10000, inplace=True)
# data['views'].where(data['views'] < 10000000, 10000000, inplace=True)

print(data['likes'].head())

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(data, data['likes']):
#     train_set = data.loc(train_index)
#     test_set = data.loc(test_index)

# print(train_set.head())

# data[['likes', 'dislikes', 'comment_count', 'views']].hist()
# plt.savefig('hists.png')
