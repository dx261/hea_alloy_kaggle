"""
compressive fracture strain of HEA modeling and prediction
"""
import os

from scipy.stats import pearsonr

from util.base_function import get_chemical_formula
from util.descriptor.magpie import get_magpie_features
from util.eval import cal_reg_metric
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import re
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

def Linear_SVR(C=1.0, gamma=0.1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="linear", C=C, gamma=gamma, epsilon=epsilon))
    ])


def RBF_SVR(C=1.0, gamma=1.0, epsilon=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon))
    ])


if __name__ == '__main__':
    dataset = pd.read_csv('./data/2_CFS_magpie_feature.csv')
    print(dataset.shape)
    dataset.drop_duplicates(keep='first', inplace=True)
    Q1 = dataset['CFS'].quantile(0.25)
    Q3 = dataset['CFS'].quantile(0.75)
    IQR = Q3 - Q1
    # 保留小于极端大的值
    dataset = dataset[dataset['CFS'] <= (Q3 + (1.5 * IQR))]
    # 保留大于极端小的值
    dataset = dataset[dataset['CFS'] >= (Q1 - (1.5 * IQR))]
    print(dataset.shape)
    Y_col = 'CFS'

    import json
    with open('config.json', 'r') as file:
        config = json.load(file)

    best_features_myy_CFS = config['best_features_myy_CFS']

    best_features_zyj_CFS = config['best_features_zyj_CFS']
    alloy_feature = pd.read_csv('./data/2_CFS_alloy_feature.csv')
    alloy_feature = alloy_feature.drop(['formula', 'CFS'], axis=1)
    # print(alloy_feature.head())

    # alloy features
    # ml_dataset = pd.concat([alloy_feature, dataset[Y_col]], axis=1).dropna()
    # X = ml_dataset[alloy_feature.columns]

    # magpie features
    ml_dataset = dataset[best_features_zyj_CFS + [Y_col]].dropna()
    X = ml_dataset[best_features_zyj_CFS]

    Y = ml_dataset[Y_col]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=48)
    model_final = RBF_SVR(C=100, gamma=0.13, epsilon=1.3)
    model_final.fit(X_train, Y_train)
    y_test_predict = model_final.predict(X_test)
    y_train_predict = model_final.predict(X_train)
    y_predict = model_final.predict(X)
    cvscore = cross_val_score(model_final, X_train, Y_train, cv=50, scoring='r2')
    evaluation_matrix = cal_reg_metric(Y_test, y_test_predict)
    evaluation_matrix_train = cal_reg_metric(Y_train, y_train_predict)
    y_cv_predict = cross_val_predict(model_final, X_train, Y_train, cv=50)
    evaluation_matrix_cv = cal_reg_metric(Y_train, y_cv_predict)
    R2 = r2_score(Y_train, y_cv_predict)
    print(cvscore, R2)
    print(evaluation_matrix_train)

    model_path = "./model/CFS_model.pkl"
    with open(model_path, 'wb') as file:
        pickle.dump(model_final, file)
    print(f"Model saved to {model_path}")

    # plot regression
    from util.plot import plot_regression_results
    plot_regression_results(Y_train, y_cv_predict, evaluation_matrix=evaluation_matrix_cv)
    plot_regression_results(Y_test, y_test_predict, Y_train, y_train_predict, plot_test_index=False,
                            evaluation_matrix=evaluation_matrix)

    # 数据优劣分类
    threshold = dataset['CFS'].median()
    dataset['CFS_category'] = dataset['CFS'].apply(lambda x: 'good_samples' if x >= threshold else 'bad_samples')
    cls_label = dataset['CFS_category']

    # pca plot in feature space
    from util.projection.PCA import PCAProjection
    pca_projection = PCAProjection()
    pca_projection.fit(X)
    pca_projection.visualize(X, cls_label, save=True, filename='./figures/pca_visualization_CFS.png')

    from util.plot import generate_shap_figures
    generate_shap_figures(model_final, X, fig_path='./figures/shap_CFS.png', n_features=3)
