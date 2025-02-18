import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def variance_threshold_selector(data, threshold=0):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    support = selector.get_support()
    dropped_features = data.columns[~support]  # ~是反运算符, 获得被筛掉的特征
    return data[data.columns[selector.get_support(indices=True)]], len(dropped_features)


def pearson_corr_feature_selection(data, threshold=0.9):
    corr_matrix = data.corr('pearson').abs()
    triangle_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # 获取上三角矩阵的索引
    to_drop = [column for column in triangle_upper.columns if any(triangle_upper[column] > threshold)]  # 找到相关性超过阈值的特征
    df_filtered = data.drop(columns=to_drop)
    return df_filtered, to_drop


if __name__ == '__main__':
    dataset = pd.read_csv("./data/2_CFS_magpie_feature.csv")
    dataset.drop_duplicates(keep='first', inplace=True)
    target = "CFS"
    Q1 = dataset[target].quantile(0.25)
    Q3 = dataset[target].quantile(0.75)
    IQR = Q3 - Q1
    # 保留小于极端大的值
    dataset = dataset[dataset[target] <= (Q3 + (1.5 * IQR))]
    # 保留大于极端小的值
    dataset = dataset[dataset[target] >= (Q1 - (1.5 * IQR))]
    dataset = dataset.drop(["formula", "composition_obj", "Weight Fraction", "Atomic Fraction", "CFS"], axis=1)
    dataset_remain, num_of_dropped = variance_threshold_selector(dataset)
    print(f"方差阈值法筛去{num_of_dropped}个特征，余{len(dataset_remain.columns)}个特征")
    dataset_remain, dropped_features_column = pearson_corr_feature_selection(dataset_remain)
    print(f"相关系数法筛去{len(dropped_features_column)}个特征，余{len(dataset_remain.columns)}个特征")
    dataset_remain.to_csv(f"./data/2_{target}_feature_selected(VarianceThreshold+PCC).csv", index=False)
