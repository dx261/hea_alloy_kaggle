"""
calculate formulas and descriptors
"""
import json
import os

import joblib
import pandas as pd

from util.alloys_features import formula_to_features
from util.descriptor.magpie import get_magpie_features
from util.base_function import get_chemical_formula

if __name__ == '__main__':
    # region phase
    dataset_name = "phase"
    dataset = pd.read_csv(os.path.join("./data/", f"1_{dataset_name}_ml_dataset.csv"))
    element_feature = dataset.columns[:-1]
    # print(element_feature)
    formulas = get_chemical_formula(dataset[element_feature])
    # print(formulas)
    df = pd.DataFrame({"formula": formulas})
    df.to_csv("./data/formula.csv", index=False)

    # generate magpie features to ｛feature_file_path｝
    feature_file_path = f"./data/2_{dataset_name}_magpie_feature.csv"
    if os.path.exists(feature_file_path):  # if the feature_file exists, skip the calculation
        print(f"skip {feature_file_path} calculation")
    else:
        df_magpie = get_magpie_features("formula.csv", data_path="./data/")
        print(df_magpie.columns)
        df_magpie.to_csv(feature_file_path, index=False)

    alloy_feature = formula_to_features(df['formula'])
    print(alloy_feature.columns)
    print(alloy_feature.shape)
    from util.preprocessing import preprocessing_dataset

    alloy_feature.fillna(0, inplace=True)
    alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)
    # endregion

    # region oxidation
    # 计算 氧化增重数据集的 物理化学 features
    dataset = pd.read_csv(os.path.join("./data/", "2_oxidation_df_composition_dataset.csv"))
    element_feature = dataset.columns
    formulas = get_chemical_formula(dataset[element_feature])
    dataset["formula"] = formulas

    # 计算氧化增重数据集（计算斜率后）的物理化学features
    dataset_name = "oxidation_slope"
    dataset = pd.read_csv(os.path.join("./data/", "oxidation_slope.csv"))
    df_magpie = get_magpie_features("formula_slope.csv", data_path="./data/", alloy_features=True)
    df_magpie['Temperature'] = dataset['temperature']
    df_magpie.to_csv('./data/2_oxidation_slope_magpie_feature.csv', index=False)
    # 合金特征计算
    alloy_feature = formula_to_features(dataset['formula'])
    alloy_feature.fillna(0, inplace=True)
    alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)
    # endregion

    # 计算硬度数据集的物理化学features
    dataset_name = "Hardness"
    dataset = pd.read_csv(os.path.join("./data/", "370composition.csv"))
    dataset['formula'] = get_chemical_formula(dataset.iloc[:, 3:-1])
    print(dataset['formula'])
    dataset.to_csv(os.path.join("./data/", "370composition.csv"))
    # 合金特征计算
    alloy_feature = formula_to_features(dataset['formula'])
    alloy_feature.fillna(0, inplace=True)
    alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)

    # magpie feature
    df = dataset['formula']
    df.to_csv("./data/formula_slope.csv", index=False)

    feature_file_path = f"./data/2_{dataset_name}_magpie_feature.csv"
    if os.path.exists(feature_file_path):  # if the feature_file exists, skip the calculation
        print(f"skip {feature_file_path} calculation")
    else:
        df_magpie = get_magpie_features("formula_slope.csv", data_path="./data/")
        print(df_magpie.columns)
        df_magpie.to_csv(feature_file_path, index=False)

    # TODO 待优化 需要精简一下代码 --zyj
    # 计算CFS数据集的特征
    # 1.magpie特征计算
    dataset_name = "CFS"
    dataset = pd.read_csv(os.path.join("./data/", "1_CFS_ml_dataset.csv"))
    # dataset["formula"] = dataset["formula"].str.replace(' ', '')
    dataset["formula"].to_csv("./data/formula_CFS.csv", index=False)
    feature_file_path = f"./data/2_{dataset_name}_magpie_feature.csv"
    if os.path.exists(feature_file_path):  # if the feature_file exists, skip the calculation
        print(f"skip {feature_file_path} calculation")
    else:
        df_magpie = get_magpie_features("formula_CFS.csv", data_path="./data/", alloy_features=True)
        df_magpie = pd.concat([df_magpie, dataset['CFS']], axis=1)  # 特征与目标合并
        df_magpie.to_csv(feature_file_path, index=False)
    # 2.合金特征计算
    CFS_alloy_feature = formula_to_features(dataset['formula'])
    CFS_alloy_feature = pd.concat([CFS_alloy_feature, dataset[['formula', 'CFS']]], axis=1)  # 特征与化学式、目标合并
    CFS_alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)

    # 计算YS数据集的特征
    dataset_name = "YS"
    dataset = pd.read_csv(os.path.join("./data/", "1_YS_ml_dataset.csv"))
    dataset["formula"].to_csv("./data/formula_YS.csv", index=False)

    # 1.magpie特征计算
    feature_file_path = f"./data/2_{dataset_name}_magpie_feature.csv"
    if os.path.exists(feature_file_path):  # if the feature_file exists, skip the calculation
        print(f"skip {feature_file_path} calculation")
    else:
        df_magpie = get_magpie_features("formula_YS.csv", data_path="./data/", alloy_features=True)
        df_magpie = pd.concat([df_magpie, dataset['YS']], axis=1)
        df_magpie.to_csv(feature_file_path, index=False)
    # 2.合金特征计算
    YS_alloy_feature = formula_to_features(dataset['formula'])
    YS_alloy_feature = pd.concat([YS_alloy_feature, dataset[['formula', 'YS']]], axis=1)
    YS_alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)
