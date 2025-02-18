import itertools
import pandas as pd


if __name__ == '__main__':
    # generate virtual space
    search_range = {"Al": [i / 100 for i in range(0, 11)],  # 范围0%-10%
                    "Ti": [i / 100 for i in range(0, 11)],  # 范围0%-10%
                    "Fe": [i / 100 for i in range(0, 101, 5)],  # 范围0%-100%
                    "Co": [i / 100 for i in range(0, 101, 5)],  # 范围0%-100%
                    "Cr": [i / 100 for i in range(0, 101, 5)],  # 范围0%-100%
                    "Ni": [i / 100 for i in range(0, 101, 5)],  # 范围0%-100%
                    }
    uniques = [i for i in search_range.values()]
    all_element_ratios = []
    for element_ratio in itertools.product(*uniques):
        if element_ratio[0]+element_ratio[1] <= 0.1 and sum(element_ratio) == 1:  # 留下Al% +Ti%的摩尔比小于10%的合金
            if 0 not in element_ratio:
                print(element_ratio)
                all_element_ratios.append(element_ratio)
    result = pd.DataFrame(all_element_ratios, columns=list(search_range.keys()))
    result.to_csv("./data/HEA_virtual_samples_simple.csv")