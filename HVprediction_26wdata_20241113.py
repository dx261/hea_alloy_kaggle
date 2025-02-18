import os
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
    data_path = "./data/"
    dataset = pd.read_csv(os.path.join(data_path, "370composition.csv"))
    Y_col = "HV"
    feature_file_path = os.path.join(data_path, "magpie_feature_hardness.csv")
    df_magpie = pd.read_csv(feature_file_path)
    dataset_all = pd.concat([dataset, df_magpie], axis=1)
    # features = elements_columns + conditions_features + magpie_features
    valence_features = ['avg s valence electrons', 'avg p valence electrons', 'avg d valence electrons',
                        'avg f valence electrons']
    features = ['MagpieData avg_dev AtomicWeight', 'MagpieData avg_dev Column',
                'MagpieData avg_dev GSvolume_pa'] + valence_features
    # print("len(features)", len(features))
    # ML 建模和评估
    ml_dataset = dataset_all[features + [Y_col]].dropna()
    ml_dataset.head()
    from sklearn.model_selection import train_test_split

    X = ml_dataset[features]
    Y = ml_dataset[Y_col]
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
    model_final = RBF_SVR(C=1642, epsilon=13.6, gamma=0.44)
    model_final.fit(X_train, Y_train)
    y_predict = model_final.predict(X_test)
    y_predict = y_predict
    y_train_predict = model_final.predict(X_train)
    y_true = Y_test
    evaluation_matrix = cal_reg_metric(y_true, y_predict)

    predata = pd.read_csv(os.path.join(data_path, "HVprediction_28w.csv"))
    # print(predata.head())
    formula = get_chemical_formula(predata)
    pre_chemistry_formula = pd.DataFrame({"formula": formula})
    pre_chemistry_formula.to_csv("./data/formula_HVprediction_28w.csv", index=False)
    predata_magpie = get_magpie_features("formula_HVprediction_28w.csv", data_path="./data/")
    feature_file_path_pre = os.path.join(data_path, "magpie_formula_HVprediction_28w.csv")
    predata_magpie.to_csv(feature_file_path_pre, index=False)
    predata_magpie = pd.read_csv(feature_file_path_pre)
    predata_all = pd.concat([predata, predata_magpie], axis=1)
    pre_X = predata_all[features].dropna()
    # pre_X.head()
    pre_Y = model_final.predict(pre_X)
    pre_Y = pd.DataFrame(pre_Y, columns=["Predicted_hardness"])
    final = pd.concat([predata, pre_Y], axis=1)
    final.to_csv("./data/HVprediction_28w_result.csv", index=False)
    print(final)
