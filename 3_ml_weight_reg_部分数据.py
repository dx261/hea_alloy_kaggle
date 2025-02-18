"""
氧化 增重回归模型
weight = composition + exposure(氧化时间) + temperature
去除突变数据

注释下面语句是全部数据的情况
dataset = dataset[(dataset["Cr"]) > 0 & (dataset["weight"] > 0.01)]
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np
import pickle
import pickle

from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import os
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score

val_config = {
}
if __name__ == '__main__':
    # 1. read data
    dataset = pd.read_csv("./data/1_oxidation_ml_dataset_modified.csv")
    tem = dataset['Temperature']
    features = pd.read_csv("./data/2_oxidation_magpie_feature.csv")
    features = features.drop(["formula", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)
    dataset_group = pd.read_csv("./data/formula_group.csv")
    Y_col = 'weight'
    print(len(dataset[Y_col]))
    print(min(dataset[Y_col]), max(dataset[Y_col]))
    print(dataset[Y_col].std())
    # filter 去除Fe-Co-Al-Ni高熵合金
    # dataset["Group"] = dataset_group["Group"]
    # dataset = dataset[(dataset["Cr"]) > 0 & (dataset["weight"] > 0.01)]
    # with plt.style.context([]):
    #     fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
    #     plt.hist(dataset[Y_col], bins=10)
    #     plt.savefig(f'./figures/hist_{Y_col}.png', bbox_inches='tight')
    # print(dataset)
    #  normalize Y by log10 transformation
    dataset[Y_col] = np.log10(dataset[Y_col])
    ml_dataset = pd.concat([dataset, features], axis=1)
    ml_feature = list(features.columns)+['Temperature']
    print(len(ml_feature))
    # region ==========ml==================
    X = ml_dataset[ml_feature]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = ml_dataset[Y_col]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=1)
    # from sklearn.feature_selection import VarianceThreshold
    # var = VarianceThreshold(threshold=0)
    # X = var.fit_transform(X)
    print(X_train.shape, Y_train.shape)
    X_train = pd.DataFrame(X_train, columns=ml_feature)
    columns_with_nulls = X_train.columns[X_train.isnull().any()]
    print(columns_with_nulls)
    # TODO feature selection?  coming soon --zyj
    feature_selector = SelectKBest(f_regression, k=len(features)).fit(X_train, Y_train)

    feature_scores = feature_selector.scores_
    print('feature_scores:', feature_scores)  # 得分越高，特征越重要
    indices = np.argsort(feature_scores)[::-1]
    val_config['feature_num'] = len(features)
    best_features = list(X_train.columns.values[indices[0:val_config['feature_num']]])
    print("best_features", best_features)
    # execute feature selection
    X_train = feature_selector.transform(X_train)
    X_test = feature_selector.transform(X_test)

    alg_dict = {
        # "Lasso": Lasso(),
        # "Ridge": Ridge(),
        # "LinearRegression": LinearRegression(),
        # 'LinearSVR': Linear_SVR(C=1),
        # 'LinearSVR2': Linear_SVR(C=100),
        # 'LinearSVR3': Linear_SVR(C=10),
        "GradientBoosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "ExtraTrees": ExtraTreesRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=4, n_jobs=4),
        "KNeighbors": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
        # 'RbfSVR': RBF_SVR(C=1),
        # 'RbfSVR1': RBF_SVR(C=10, gamma=0.20),
        # 'RbfSVR2': RBF_SVR(C=100, gamma=0.10),
        # 'RbfSVR3': RBF_SVR(C=1000, gamma=0.05),
        # 'RbfSVR4': RBF_SVR(C=0.1, gamma=0.01),
    }
    # iterate algorithms to get model with best score by leave one group cross validation
    best_model = None
    best_score = -10 ** 10
    y_predict = None
    for alg_name in alg_dict.keys():
        model = alg_dict[alg_name]
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        # score = -r2_score(Y_test, y_predict)
        # score = - mean_squared_error(Y_test, y_predict) ** 0.5
        mse = mean_squared_error(Y_test, y_predict)
        print(f"mse: {mse}")
        # leave one group cross validation
        score = (-np.mean(
            cross_val_score(model, X, Y, scoring='neg_mean_squared_error'))) ** 0.5

        y_cv_predict = cross_val_predict(model, X, Y, cv=10)
        cvR2 = r2_score(Y, y_cv_predict)
        print(f"{alg_name} {cvR2}")
        if cvR2 > best_score:
            best_model = model
            best_score = cvR2
    # save the best model
    print(f"best score {best_score} best model {best_model}")
    model_final = best_model
    # X_df = pd.DataFrame(feature_selection.transform(X), columns=best_features)
    # model_final.fit(X_df, Y)
    y_cv_predict = cross_val_predict(best_model, X, Y, cv=10)
    dataset[f'{Y_col}_predict'] = y_cv_predict

    with plt.style.context([]):
        x = Y_col
        y = f'{Y_col}_predict'
        r2 = r2_score(dataset[x], dataset[y])
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.scatter(dataset[x], dataset[y], c='blue', alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        #
        lim_max = max(max(dataset[x]), max(dataset[y]))
        lim_min = min(min(dataset[x]), min(dataset[y]))
        scale_min = lim_min - (lim_max - lim_min) * 0.1
        scale_max = lim_max + (lim_max - lim_min) * 0.1
        plt.plot([scale_min, scale_max], [scale_min, scale_max], color='black', linestyle="--")
        # plt.xticks(fontsize=12, fontweight='bold')
        # plt.yticks(fontsize=12, fontweight='bold')
        # plt.xlabel(x, fontsize=12, fontweight='bold')
        # plt.ylabel(y, fontsize=12, fontweight='bold')
        plt.xlim(scale_min, scale_max)
        plt.ylim(scale_min, scale_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        # plt.savefig(f'./figures/R2_lower5.png', bbox_inches='tight')
        plt.show()
# endregion ============== ml ==================
