"""
data preprocessing
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from util.base_function import get_chemical_formula


def set_group_for_oxidation():
    """
    set group number for each alloy with same composition
    """
    dataset = pd.read_csv("./data/1_oxidation_ml_dataset.csv")
    ele_col = list(dataset.columns[:-3])  # 元素列表
    print("elements", ele_col)
    # 计算化学式
    formulas = get_chemical_formula(dataset[ele_col])
    dataset["formula"] = formulas  # 添加成分列
    print(formulas)
    df = pd.DataFrame({"formula": list(set(formulas))})
    df["Group"] = list(df.index)  # 加索引
    print(df.head(30))
    df_all = pd.merge(dataset, df, on="formula")  # 把索引合到dataset里
    print(df_all.head(30))
    df_all.to_csv("./data/formula_group.csv", index=False)
    print("set_group_for_oxidation done")

def calculate_oxidation_slope():
    """
    calculate oxidation slope for each group
    """
    df = pd.read_csv("./data/1_oxidation_ml_dataset_modified.csv")
    ele_col = list(df.columns[:-4])  # 元素列表
    formulas = get_chemical_formula(df[ele_col])
    df["formula"] = formulas
    # print(df)
    # df.to_csv("./data/1_oxidation_ml_dataset_modified.csv", index=False)
    grouped = df.groupby(by=["formula", "Temperature"])  # 根据多个元素分组
    formula_slope = []
    tem = []
    for i in range(len(list(grouped.groups.keys()))):
        formula_slope.append(list(grouped.groups.keys())[i][0])
        tem.append(list(grouped.groups.keys())[i][1])
    print(formula_slope, tem)
    print(len(formula_slope), len(tem))
    slope = []
    for i in range(len(list(grouped.groups.keys()))):
        group_a = grouped.get_group((formula_slope[i], tem[i]))  # 对多元素分组获取每一组时需要传入元组
        model = LinearRegression()  # 用线性回归拟合并计算斜率
        Y = group_a["weight"].to_frame()  # 提取单列需要额外转成dataframe
        X = group_a["Exposure"].to_frame()
        model.fit(X, Y)
        slope.append(float(model.coef_[0][0]))  # 获得斜率信息
    print(formula_slope, tem)
    print(f"slopes: {slope}")
    df_slope = pd.DataFrame({"formula": formula_slope, "slope": slope, "temperature": tem})
    df_slope.to_csv("./data/oxidation_slope.csv", index=False)


if __name__ == '__main__':
    # set_group_for_oxidation()
    calculate_oxidation_slope()
