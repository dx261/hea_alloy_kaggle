import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json

if __name__ == "__main__":
    CFS_model = joblib.load("./model/CFS_model.pkl")
    YS_model = joblib.load("./model/YS_model.pkl")
    HV_model = joblib.load("./model/hardness_model.pkl")
    phase_scaler = joblib.load("./model/phase_scaler.pkl")
    Phase_model = joblib.load("./model/phase_model.pkl")
    dataset_virtual_samples_magpie = pd.read_csv("./data/2_virtual_samples_magpie_feature.csv")
    dataset_virtual_samples_alloy = pd.read_csv("./data/2_virtual_samples_alloy_feature.csv")
    ml_dataset = pd.concat([dataset_virtual_samples_magpie, dataset_virtual_samples_alloy.iloc[:, :-1]], axis=1)
    ml_dataset = ml_dataset.drop(['composition_obj'], axis=1)
    dataset_virtual_samples_magpie = dataset_virtual_samples_magpie.drop(["formula", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)

    with open('config.json', 'r') as file:
        config = json.load(file)

    CFS_features = config['best_features_zyj_CFS']
    HV_features = config['HV_features']
    YS_features = config['best_features_zyj_YS']

    # CFS_prediction
    X_CFS = ml_dataset[CFS_features]
    CFS_prediction = CFS_model.predict(X_CFS)
    # YS_prediction
    X_YS = dataset_virtual_samples_alloy.iloc[:, :-1]
    YS_prediction = YS_model.predict(X_YS)
    # HV_prediction
    X_HV = ml_dataset[HV_features]
    HV_prediction = HV_model.predict(X_HV)
    # Phase_prediction
    X_Phase = dataset_virtual_samples_magpie
    X_Phase_std = phase_scaler.transform(X_Phase)
    Phase_prediction = Phase_model.predict(X_Phase_std)

    result = pd.DataFrame({"formula": dataset_virtual_samples_alloy['formula'], "CFS_prediction": CFS_prediction,
                           "YS_prediction": YS_prediction, "HV_prediction": HV_prediction,
                           "Phase_prediction": Phase_prediction})
    print(result.head(5))
    result.to_csv("./data/3_virtual_samples_prediction.csv", index=False)

    # PCA

