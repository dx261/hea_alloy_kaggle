"""
phase prediction for HEA
use hyperopt to perform hyperparameters optimization
"""
import seaborn as sns
import pickle
from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import os
import scipy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

val_config = {}


def read_data():
    """化学组成"""
    dataset = pd.read_csv("./data/1_phase_ml_dataset.csv")
    Y_col = 'Phase'
    features = list(dataset.columns[:-1])
    return dataset, Y_col, features


def read_magpie_data():
    """
    使用化学式计算的物理化学特征
    :return:
    """
    dataset = pd.read_csv("./data/2_phase_magpie_feature.csv")
    Y_col = 'Phase'
    features = list(dataset.columns[2:])
    return dataset, Y_col, features


def read_alloy_data():
    """
    使用 23 个特征
    :return:
    """
    dataset = pd.read_csv("./data/2_phase_alloy_feature.csv")
    Y_col = 'Phase'
    features = list(dataset.columns)
    return dataset, Y_col, features


def objective(params, alg_name, X_train, Y_train, X_test, Y_test):
    """
    超参数优化
    """
    model = alg_dict[alg_name](**params)
    model.fit(X_train, Y_train)
    y_predict = model.predict(X_test)
    score = accuracy_score(Y_test, y_predict)
    return {'loss': -score, 'status': STATUS_OK}


if __name__ == '__main__':
    # dataset, Y_col, features = read_data()
    dataset, Y_col, features = read_magpie_data()
    # dataset, Y_col, features = read_alloy_data()
    print(max(dataset))
    # use magpie features
    print(f"features:{features}")
    ml_dataset = dataset
    # ml
    X = ml_dataset[features]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = pd.read_csv("./data/1_phase_ml_dataset.csv")[Y_col]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)

    print(X_train.shape, Y_train.shape)
    X_train = pd.DataFrame(X_train, columns=features)
    columns_with_nulls = X_train.columns[X_train.isnull().any()]
    print(columns_with_nulls)
    feature_selection = SelectKBest(f_classif, k=len(features)).fit(X_train, Y_train)

    feature_scores = feature_selection.scores_
    print('feature_scores:', feature_scores)
    indices = np.argsort(feature_scores)[::-1]
    val_config['feature_num'] = len(features)
    best_features = list(X_train.columns.values[indices[0:val_config['feature_num']]])
    print("best_features", best_features)
    X_train = feature_selection.transform(X_train)
    X_test = feature_selection.transform(X_test)
    sc = MinMaxScaler()
    alg_dict = {
        # "LogisticRegression": LogisticRegression,
        "DecisionTree": DecisionTreeClassifier,
        "KNeighbors": KNeighborsClassifier,
        "RandomForest": RandomForestClassifier,
        "ExtraTrees": ExtraTreesClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "AdaBoost": AdaBoostClassifier,
        "SVC": SVC,
    }
    best_model = None
    best_score = 0
    best_y_predict = None
    # use hyperopt to search in search_space
    search_space = {
        # "LogisticRegression": {
        #     "C": hp.loguniform('C', -4, 4),
        # },
        "KNeighbors": {
            "n_neighbors": hp.randint("n_neighbors", 1, 20),
        },
        "DecisionTree": {
            # "criterion": hp.choice('criterion', ['gini', 'entropy']),
            "max_depth": hp.randint("max_depth", 2, 10),
        },
        "RandomForest": {
            "n_estimators": hp.randint('n_estimators', 50, 150),
            "min_samples_split": hp.randint('min_samples_split', 2, 10),
            "max_depth": hp.randint("max_depth", 2, 10),
        },
        "ExtraTrees": {
            "n_estimators": hp.randint('n_estimators', 50, 150),
            "min_samples_split": hp.randint('min_samples_split', 2, 10),
            "max_depth": hp.randint("max_depth", 2, 10),
        },
        "GradientBoosting": {
            "n_estimators": hp.randint('n_estimators', 10, 50),
            "learning_rate": hp.loguniform('learning_rate', -4, 0),
            "max_depth": hp.randint("max_depth", 2, 10),
        },
        "AdaBoost": {
            "n_estimators": hp.randint('n_estimators', 10, 50),
            "learning_rate": hp.loguniform('learning_rate', -4, 0),
        },
        "SVC": {
            "C": hp.loguniform('C', -1000, 1000),
            # "kernel": hp.choice('kernel', ['linear', 'rbf', 'poly']),
            # "gamma": hp.choice('gamma', ['scale', 'auto']),
        }
    }
    for alg_name in alg_dict.keys():
        hp_space = search_space[alg_name]
        print(hp_space)
        best = fmin(fn=lambda params: objective(params, alg_name, X_train, Y_train, X_test, Y_test),
                    space=hp_space,
                    algo=tpe.suggest,
                    max_evals=50)
        print(" best params", alg_name, best)
        model = alg_dict[alg_name](**best)
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        score = accuracy_score(Y_test, y_predict)
        print(f"{alg_name} {score}")
        if score > best_score:
            best_model = model
            best_score = score
            best_y_predict = y_predict.copy()
    # save the best model
    print(f"best accuracy_score {best_score} best model {best_model}")
