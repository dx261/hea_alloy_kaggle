"""
phase prediction for HEA
"""
import seaborn as sns
import pickle
from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import scipy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

val_config = {}


def read_data():
    dataset = pd.read_csv("./data/1_phase_ml_dataset.csv")  # 最后一列是相信息，前面是成分特征
    Y_col = 'Phase'
    features = list(dataset.columns[:-1])
    return dataset, Y_col, features


def read_magpie_data():
    """
    使用化学式特征
    :return:
    """
    dataset = pd.read_csv("./data/2_phase_magpie_feature.csv")
    if "Weight Fraction" in dataset.columns:
        dataset1 = dataset.drop(["formula", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)
    else:
        dataset1 = dataset.drop(["formula", "composition_obj"], axis=1)
    Y_col = 'Phase'
    features = list(dataset1.columns)
    return dataset1, Y_col, features


def read_alloy_data():
    """
    使用23个合金特征
    :return:
    """
    dataset = pd.read_csv("./data/2_phase_alloy_feature.csv")
    Y_col = 'Phase'
    features = list(dataset.columns)
    return dataset, Y_col, features

def RBF_SVC(C=1.0, gamma=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=C, gamma=gamma))
    ])


if __name__ == '__main__':
    # use different feature subset

    # dataset, Y_col, features = read_data()        # use element%
    dataset, Y_col, features = read_magpie_data()   # use magpie features
    # dataset, Y_col, features = read_alloy_data()  # use alloy features

    print(f"features:{features}")
    ml_dataset = dataset
    # ml
    X = ml_dataset[features]
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)

    # TODO 想搞个pipeline但是建模有问题了 --zyj
    scaler_path = "./model/phase_scaler.pkl"
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"Model saved to {scaler_path}")
    Y = pd.read_csv("./data/1_phase_ml_dataset.csv")[Y_col]

    # 为二分类重构目标
    binary_classification = False
    if binary_classification:  # 改为二分类
        for i in range(len(Y)):
            if Y.iloc[i] != 'FCC':
                Y.iloc[i] = 'NOT_FCC'

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)  # 要对FCC预测效果好

    # print(X_train.shape, Y_train.shape)
    X_train = pd.DataFrame(X_train, columns=features)
    columns_with_nulls = X_train.columns[X_train.isnull().any()]
    print(columns_with_nulls)
    k_sample = len(features)
    feature_selection = SelectKBest(f_classif, k=k_sample).fit(X_train, Y_train)
    feature_scores = feature_selection.scores_
    indices = np.argsort(feature_scores)[::-1] # -1转升序为降序，返回索引
    val_config['feature_num'] = k_sample  # 一个字典
    best_features = list(X_train.columns.values[indices[0:val_config['feature_num']]])
    print("best_features", best_features)
    X_train = feature_selection.transform(X_train)  # 提取筛选出来的k个特征
    X_test = feature_selection.transform(X_test)
    alg_dict = {
        # "LogisticRegression": LogisticRegression(),
        # "KNeighbors": KNeighborsClassifier(),
        # "DecisionTree": DecisionTreeClassifier(),
        # "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=4, n_jobs=-1),
        # "ExtraTrees": ExtraTreesClassifier(
        #     **{'max_depth': 5, 'min_samples_split': 6, 'n_estimators': 133}),
        # "GradientBoosting": GradientBoostingClassifier(),
        # "AdaBoost": AdaBoostClassifier(),
        "SVC": SVC(C=16103.081755889294),
    }
    best_model = None
    best_score = 0
    best_y_predict = None
    # 选择测试集最优算法（数据较多时 可以不使用交叉验证）
    for alg_name in alg_dict.keys():
        model = alg_dict[alg_name]
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

    scaler_path = "model/Phase_model.pkl"
    with open(scaler_path, 'wb') as file:
        pickle.dump(best_model, file)
    print(f"Model saved to {scaler_path}")

    # dataset['Y_predict'] = best_y_predict
    class_labels = list(set(Y_test))
    cm = confusion_matrix(Y_test, best_y_predict, labels=class_labels)
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'./figures/confusion_matrix.png', bbox_inches='tight')
    plt.show()

    from util.plot import generate_shap_figures
    generate_shap_figures(best_model, X, fig_path='./figures/shap_Phase.png', n_features=1)

    # use best model to predict
    # model_final = best_model
    # model_final.fit(X, Y)
    # dataset_predict = pd.read_csv("./data/2_oxidation_slope_magpie_feature.csv")
    # y_predict = model_final.predict(scaler.transform(dataset_predict[features]))
    # print(y_predict)
    # dataset_predict.insert(0, "phase_predict", y_predict)
    # dataset_predict.to_csv("./data/3_oxidation_phase_predict.csv", index=False)

    # PCA projection
    from util.projection.PCA import PCAProjection
    pca_projection = PCAProjection()
    pca_projection.fit(X)
    pca_projection.visualize(X, Y, save=True, filename='./figures/pca_visualization_phase.png')
    # mapping the virtual samples
    dataset_virtual_samples_magpie = pd.read_csv("./data/2_virtual_samples_magpie_feature.csv")
    dataset_virtual_samples_magpie = dataset_virtual_samples_magpie.drop(["formula", "composition_obj", "Atomic Fraction", "Weight Fraction"], axis=1)

    X_virtual = dataset_virtual_samples_magpie
    Y_virtual = ["virtual"] * len(dataset_virtual_samples_magpie)
    X_virtual = scaler.transform(X_virtual)
    X_all = np.concatenate((X, X_virtual), axis=0)
    Y_all = np.concatenate((Y, Y_virtual), axis=0)
    pca_projection.visualize(X_all, Y_all, save=True, filename='./figures/pca_visualization_phase_all.png')