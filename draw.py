"""
draw figures
"""
import matplotlib.pyplot as plt
import pandas as pd
from util.base_function import get_chemical_formula
if __name__ == '__main__':
    # region ===visualize each oxidation curve=======
    df_all = pd.read_csv("./data/formula_group.csv")
    for i in df_all["Group"]:
        print("Group:", i)
        df_group = df_all[df_all["Group"] == i]
        plt.scatter(df_group["Exposure"], df_group["weight"], c="red")
        plt.xlabel("Exposure")
        plt.ylabel("Weight")
        plt.title(set(df_group.formula))
        plt.show()
    # endregion ======================================


