"""
hardness of HEA modeling and prediction
"""
import os

from scipy.stats import pearsonr

from util.base_function import get_chemical_formula
from util.descriptor.magpie import get_magpie_features
from util.eval import cal_reg_metric
from util.plot import plot_regression_results
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
    # 1. 读取高熵合金硬度数据集
    data_path = "./data/"
    dataset = pd.read_csv(os.path.join(data_path, "370composition.csv"))
    print("rows", dataset.shape[0])
    Y_col = 'HV'
    elements_columns = list(dataset.columns[3:])
    dataset.head()
    chemical_formula_list = dataset['formula']
    df_chemistry_formula = pd.DataFrame({"formula": chemical_formula_list, "target": dataset[Y_col]})
    df_chemistry_formula.to_csv("./data/formula_hardness.csv", index=False)
    # if 特征已经计算过，不需重复计算
    feature_file_path = os.path.join(data_path, "magpie_feature_hardness.csv")
    if os.path.exists("./data/magpie_feature_hardness.csv"):
        df_magpie = pd.read_csv(feature_file_path)
    else:
        df_magpie = get_magpie_features("formula_hardness.csv", data_path="./data/")
        df_magpie.to_csv(feature_file_path, index=False)
        print(f"save features to {feature_file_path}")
    dataset_all = pd.concat([dataset, df_magpie], axis=1)
    print(dataset_all.head(5))

    import json
    with open('config.json', 'r') as file:
        config = json.load(file)

    features = config['HV_features']
    alloy_features = pd.read_csv("./data/2_Hardness_alloy_feature.csv")
    Tm = alloy_features['Melting temperature']

    # ML 建模和评估
    ml_dataset = pd.concat([Tm, dataset_all], axis=1).dropna()

    from sklearn.model_selection import train_test_split

    X = ml_dataset[features]
    Y = ml_dataset[Y_col]
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=6)
    model_final = RBF_SVR(C=1642, epsilon=13.6, gamma=0.44)
    model_final.fit(X_train, Y_train)
    y_predict = model_final.predict(X_test)
    y_train_predict = model_final.predict(X_train)
    y_true = Y_test
    evaluation_matrix = cal_reg_metric(y_true, y_predict)
    cvscore = cross_val_score(model_final, X_train, Y_train, cv=10)
    y_cv_predict = cross_val_predict(model_final, X_train, Y_train, cv=10)
    R, _ = pearsonr(Y_train, y_cv_predict)
    print(cvscore, np.mean(cvscore), R.round(3))
    evaluation_matrix_cv = cal_reg_metric(Y_train, y_cv_predict)

    model_path = "./model/hardness_model.pkl"
    with open(model_path, 'wb') as file:
        pickle.dump(model_final, file)
    print(f"Model saved to {model_path}")
    # plot regression
    # cv
    plot_regression_results(Y_train, y_cv_predict, evaluation_matrix=evaluation_matrix_cv)
    # test set
    save_path = f'./figures/HEA_hardness_reg.png'
    plot_regression_results(Y_test, y_predict, Y_train, y_train_predict, evaluation_matrix=evaluation_matrix,
                            save_path=save_path)

    from util.plot import generate_shap_figures
    generate_shap_figures(model_final, X, fig_path='./figures/shap_hardness.png', n_features=3)
