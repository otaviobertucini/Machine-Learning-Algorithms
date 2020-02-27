from utils.utils import open_cvs
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from pandas.plotting import scatter_matrix


def prepare_data():

    data = open_cvs('/home/otavio/ml/datasets/wine-quality/winequality-red.csv')

    data['fa_split'] = np.ceil(data['fixed acidity'])
    data['ph_acidity'] = (data['pH']*data['volatile acidity'])/(data['alcohol'])
    data['alcohol_acidity'] = (data['alcohol']/data['volatile acidity'])
    data['citric_acidity'] = data['citric acid'] / data['volatile acidity']
    data['alcohol_sweet'] = data['alcohol'] * data['fixed acidity']

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data['fa_split']):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]

    for sett in (train_set, test_set):
        sett.drop(['fa_split'], axis=1, inplace=True)
        sett.drop(['fixed acidity'], axis=1, inplace=True)
        sett.drop(['residual sugar'], axis=1, inplace=True)
        sett.drop(['free sulfur dioxide'], axis=1, inplace=True)
        sett.drop(['pH'], axis=1, inplace=True)
        sett.drop(['chlorides'], axis=1, inplace=True)
        sett.drop(['density'], axis=1, inplace=True)
        sett.drop(['total sulfur dioxide'], axis=1, inplace=True)

    # data = train_set.copy()
    # corr = data.corr()
    # print(corr['quality'].sort_values(ascending=False))

    # attributes = ['quality', 'alcohol', 'sulphates', 'citric acid', 'volatile acidity']
    # scatter_matrix(data[attributes])

    # train_set.hist()
    # plt.savefig('hist.png')

    train_data = train_set.drop('quality', axis=1)
    labels = train_set['quality'].copy()

    print("Linear regression")

    lin_reg = LinearRegression()
    lin_reg.fit(train_data, labels)

    some_data = train_data.iloc[:5]
    some_labels = labels.iloc[:5]

    print("Predictions:\t", lin_reg.predict(some_data))
    print("Real:\t", list(some_labels))

    predictions = lin_reg.predict(train_data)
    lin_mse = mean_squared_error(labels, predictions)
    lin_rmse = np.sqrt(lin_mse)
    print('MSE: ' + str(lin_rmse))

    print("--------------------------")

    print("Decision tree")
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(train_data, labels)

    predictions_tree = tree_reg.predict(train_data)
    tree_mse = mean_squared_error(labels, predictions_tree)
    tree_rmse = np.sqrt(tree_mse)
    print('MSE: ' + str(tree_rmse))

    scores = cross_val_score(tree_reg, train_data, labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)

    print("Scores:", rmse_scores)
    print("Mean:", rmse_scores.mean())
    print("STD: ", rmse_scores.std())

    print("--------------------------")

    print("Random forest")
    forest_reg = RandomForestRegressor()
    forest_reg.fit(train_data, labels)

    predictions_forest = forest_reg.predict(train_data)
    forest_mse = mean_squared_error(labels, predictions_forest)
    forest_rmse = np.sqrt(forest_mse)
    print('MSE: ' + str(forest_rmse))

    scores_forest = cross_val_score(forest_reg, train_data, labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores_forest = np.sqrt(-scores_forest)

    print("Scores:", rmse_scores_forest)
    print("Mean:", rmse_scores_forest.mean())
    print("STD: ", rmse_scores_forest.std())

    print("--------------------------")


prepare_data()
