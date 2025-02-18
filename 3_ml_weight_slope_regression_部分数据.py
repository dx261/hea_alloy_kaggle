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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np
import pickle
from scipy import stats

from util.eval import cal_reg_metric
from util.plot import plot_regression_results

# slope数据量太少了 效果很差
if __name__ == '__main__':
    dataset_slope_original = pd.read_csv("./data/2_oxidation_slope_magpie_feature.csv")
    dataset_slope = pd.read_csv("./data/oxidation_slope.csv")
    dataset_slope_original = dataset_slope_original.drop(["formula", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)
    Y_col = 'slope'
    feature_names = list(dataset_slope_original.columns)
    # print(dataset_slope_original[Y_col].std())  # 计算标准差
    # with plt.style.context([]):  # 传递一个空列表 [] 表示不应用任何预定义的样式表，恢复到默认样式。可以传入字典来规定绘图的参数/样式
    #     fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
    #     plt.hist(dataset_slope_original[Y_col], bins=10)
        # plt.show()
    X = dataset_slope_original[feature_names]
    # print(type(X))
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    Y = dataset_slope[Y_col]
    # print(type(X))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=1)
    k_best = len(feature_names)
    feature_selector = SelectKBest(f_regression, k=k_best).fit(X_train, Y_train)
    feature_scores = feature_selector.scores_
    # print('feature_scores:', feature_scores)  # 得分越高，特征越重要
    indices = np.argsort(feature_scores)[::-1]
    val_config = {}
    val_config['feature_num'] = k_best
    best_features = list(X_train.columns.values[indices[0:val_config['feature_num']]])
    # print("best_features", best_features)
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
        "GradientBoosting": GradientBoostingRegressor(random_state=4),
        "AdaBoost": AdaBoostRegressor(random_state=4),
        "ExtraTrees": ExtraTreesRegressor(random_state=4),
        "RandomForest": RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=4, n_jobs=4),
        # "KNeighbors": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(random_state=4),
        # "SVR_rbf": SVR(C=1000, gamma=0.05, kernel='rbf'),
        # "SVR_linear": SVR(C=1000, gamma=0.05, kernel='linear'),
        # 'RbfSVR': RBF_SVR(C=1),
        # 'RbfSVR1': RBF_SVR(C=10, gamma=0.20),
        # 'RbfSVR2': RBF_SVR(C=100, gamma=0.10),
        # 'RbfSVR3': RBF_SVR(C=1000, gamma=0.05),
        # 'RbfSVR4': RBF_SVR(C=0.1, gamma=0.01),
    }
    best_model = None
    best_score = -10**10
    y_predict = None
    for alg_name in alg_dict.keys():
        model = alg_dict[alg_name]
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        r2 = r2_score(Y_test, y_predict)
        mse = mean_squared_error(Y_test, y_predict)
        # print(f"test_mse: {mse}\t test_r2: {r2}\t model:{model}")
        score = cross_val_score(model, X, Y, cv=10, scoring='r2')
        cv_predict = cross_val_predict(model, X, Y, cv=10)
        evaluation_matrix = cal_reg_metric(Y, cv_predict)
        print(evaluation_matrix)
        avg_score = evaluation_matrix.get('R2')
        print(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
    # save the best model
    print(f"best score {best_score} best model {best_model}")
    all_predict = best_model.predict(X)
    cv_predict = cross_val_predict(best_model, X, Y, cv=10)
    evaluation_matrix = cal_reg_metric(Y, cv_predict)
    print(evaluation_matrix)
    plot_regression_results(Y, cv_predict, evaluation_matrix=evaluation_matrix)