import os

import pandas as pd
from util.base_function import get_chemical_formula
from util.alloys_features import formula_to_features, find_elements
from util.descriptor.magpie import get_magpie_features

if __name__ == '__main__':
    # dataset = pd.read_csv(os.path.join("./data/", "HEA_virtual_samples_simple.csv"))
    # element_feature = dataset.columns[1:]
    # # print(element_feature)
    # formulas = get_chemical_formula(dataset[element_feature])
    # # print(formulas)
    # df = pd.DataFrame({"formula": formulas})
    # df.to_csv("./data/formula_virtual_samples.csv", index=False)
    dataset = pd.read_csv(os.path.join("./data/", "formula_virtual_samples.csv"))
    dataset_name = 'virtual_samples'
    skip_magpie = False
    if not skip_magpie:  # not false不跳过计算
        df_magpie = get_magpie_features("formula_virtual_samples.csv", data_path="./data/")
        df_magpie.to_csv(f"./data/2_{dataset_name}_magpie_feature.csv", index=False)
    # 2.合金特征计算
    YS_alloy_feature = formula_to_features(dataset['formula'])
    YS_alloy_feature = pd.concat([YS_alloy_feature, dataset['formula']], axis=1)
    YS_alloy_feature.to_csv(f"./data/2_{dataset_name}_alloy_feature.csv", index=False)
