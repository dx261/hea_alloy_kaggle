"""
predict training dataset
"""
import json
import os

import joblib
import pandas as pd

from util.alloys_features import formula_to_features, formula_to_ratio_dataset
from util.base_function import get_chemical_formula
from util.descriptor.magpie import get_magpie_features

if __name__ == '__main__':
    # 1. get all formulas
    dataset_name = "Hardness"
    dataset = pd.read_csv(os.path.join("./data/", "370composition.csv"))
    print(dataset['formula'])
    dataset_CFS = pd.read_csv(os.path.join("./data/", "1_CFS_ml_dataset.csv"))
    print(dataset_CFS["formula"])
    dataset_YS = pd.read_csv(os.path.join("./data/", "1_YS_ml_dataset.csv"))
    print(dataset_YS["formula"])

    all_formula = pd.concat([dataset['formula'], dataset_CFS["formula"], dataset_YS["formula"]])
    print(all_formula)
    # 1.magpie特征计算
    all_formula.to_csv("./data/formula_all.csv",index=False)
    feature_file_path = f"./data/2_all_formula_magpie_feature.csv"
    if os.path.exists(feature_file_path):  # if the feature_file exists, skip the calculation
        print(f"skip {feature_file_path} calculation")
        df_magpie = pd.read_csv(feature_file_path)
    else:
        df_magpie = get_magpie_features("formula_all.csv", data_path="./data/", alloy_features=True)
        df_magpie.to_csv(feature_file_path, index=False)
    # 2.合金特征计算
    alloy_feature = formula_to_features(dataset['formula'])
    alloy_feature.to_csv(f"./data/2_formula_all_alloy_feature.csv", index=False)

    # 3. use model to predict
    CFS_model = joblib.load("./model/CFS_model.pkl")
    YS_model = joblib.load("./model/YS_model.pkl")
    HV_model = joblib.load("./model/hardness_model.pkl")
    phase_scaler = joblib.load("./model/phase_scaler.pkl")
    Phase_model = joblib.load("./model/phase_model.pkl")
    with open('config.json', 'r') as file:
        config = json.load(file)
    CFS_features = config['best_features_zyj_CFS']
    HV_features = config['HV_features']
    YS_features = config['best_features_zyj_YS']
    alloy_feature = pd.read_csv('./data/2_YS_alloy_feature.csv')
    alloy_feature = alloy_feature.drop(['formula', 'YS'], axis=1)
    print(alloy_feature.columns)
    X = pd.concat([df_magpie, alloy_feature], axis=1)
    print(X.shape)

    #print(X.columns)
    #X = X.fillna(0)
    y1 = CFS_model.predict(X[CFS_features])
    print(y1.shape)
    #print(X[HV_features])
    #y2 = YS_model.predict(X[list(alloy_feature.columns)])
    y3 = HV_model.predict(X[HV_features].fillna(0))
    #print(y2.shape)
    print(y3.shape)
    df_all, element_columns = formula_to_ratio_dataset(pd.DataFrame(all_formula))
    df_all["CFS"] = y1
    df_all["HV"] = y3
    df_all.to_csv("./data/6_training_prediction.csv", index=False)
