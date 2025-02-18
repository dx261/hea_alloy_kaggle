"""
Yield strength of HEA modeling and prediction
"""
from util.eval import cal_reg_metric

import pandas as pd
import numpy as np
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
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

from util.plot import plot_regression_results


def lgbm_model(boosting_type='gbdt', objective='regression', learning_rate=0.15):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", lgb.LGBMRegressor(boosting_type=boosting_type, objective=objective, learning_rate=learning_rate))
    ])


if __name__ == '__main__':
    dataset = pd.read_csv("./data/2_YS_magpie_feature.csv")
    print(dataset.shape)
    dataset.drop_duplicates(keep='first', inplace=True)
    Q1 = dataset['YS'].quantile(0.25)
    Q3 = dataset['YS'].quantile(0.75)
    IQR = Q3 - Q1
    # 保留小于极端大的值
    dataset = dataset[dataset['YS'] <= (Q3 + (1.5 * IQR))]
    # 保留大于极端小的值
    dataset = dataset[dataset['YS'] >= (Q1 - (1.5 * IQR))]
    print(dataset.shape)
    Y_col = 'YS'

    alloy_feature = pd.read_csv('./data/2_YS_alloy_feature.csv')
    alloy_feature = alloy_feature.drop(['formula', 'YS'], axis=1)
    print(alloy_feature.columns)

    # magpie features
    # ml_dataset = dataset[best_features_zyj + [Y_col]].dropna()
    # X = ml_dataset[best_features_zyj]

    # alloy features
    ml_dataset = pd.concat([alloy_feature, dataset[Y_col]], axis=1).dropna()
    X = ml_dataset[alloy_feature.columns]

    Y = ml_dataset[Y_col]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=98)
    model_final = lgbm_model(boosting_type='gbdt', objective='regression', learning_rate=0.15)
    model_final.fit(X_train, Y_train)
    y_test_predict = model_final.predict(X_test)
    y_train_predict = model_final.predict(X_train)
    y_predict = model_final.predict(X)
    cvscore = cross_val_score(model_final, X_train, Y_train, cv=10, scoring='r2')
    evaluation_matrix = cal_reg_metric(Y_test, y_test_predict)
    evaluation_matrix_train = cal_reg_metric(Y_train, y_train_predict)
    y_cv_predict = cross_val_predict(model_final, X_train, Y_train, cv=10)
    R2 = r2_score(Y_train, y_cv_predict)
    RMSE = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    print(RMSE, R2)
    print(evaluation_matrix_train)
    evaluation_matrix_cv = cal_reg_metric(Y_train, y_cv_predict)

    model_path = "./model/YS_model.pkl"
    with open(model_path, 'wb') as file:
        pickle.dump(model_final, file)
    print(f"Model saved to {model_path}")
    # plot test set
    # cv
    plot_regression_results(Y_train, y_cv_predict, evaluation_matrix=evaluation_matrix_cv)
    save_path = f'./figures/HEA_YS_reg.png'
    plot_regression_results(Y_test, y_test_predict, Y_train, y_train_predict, evaluation_matrix=evaluation_matrix
                            , save_path=save_path)

    # 数据优劣分类
    threshold = dataset['YS'].median()
    dataset['YS_category'] = dataset['YS'].apply(lambda x: 'good_samples' if x >= threshold else 'bad_samples')
    cls_label = dataset['YS_category']

    # pca plot in feature space
    from util.projection.PCA import PCAProjection
    pca_projection = PCAProjection()
    pca_projection.fit(X)
    pca_projection.visualize(X, cls_label, save=True, filename='./figures/pca_visualization_YS.png')

    from util.plot import generate_shap_figures
    generate_shap_figures(model_final, X, fig_path='./figures/shap_YS.png', n_features=3)