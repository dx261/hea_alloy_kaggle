"""
data explore analysis
"""

# 氧化数据分析
import pandas as pd
df = pd.read_excel('./data/Oxidation data.xlsx')
# 数据量
print(df.shape)
print(df.columns)
print(df.head(10))
df.to_csv("./data/1_oxidation_ml_dataset.csv", index=False)
""" 15种元素 + 测试温度 + 暴露时间 = 氧化增重（随着时间）
氧化增重是看最后的收敛的重量吗？
"""

df = pd.read_excel('./data/Phase data.xlsx')
print(df.shape)
print(df.columns)
print(df.head(10))
# 填充空缺值
df.fillna(0, inplace=True)
# 填充后数据集
print(df.head(10))

# 对Phase 进行频数统计
print(df['Phase'].value_counts())
"""
BCC 和 FCC 和 BCC + FCC 和 FCC+BCC 最多，其他 数据类别存在一定的不均衡。
"""
df.to_csv("./data/1_phase_ml_dataset.csv", index=False)

# 数据预处理
df = pd.read_excel('./data/Oxidation data.xlsx')
element_cols = df.columns[:-3]
# 提取实验数据的 composition
df_composition = df[element_cols].drop_duplicates()
df_composition.to_csv("./data/2_oxidation_df_composition_dataset.csv", index=False)

# vis 氧化增重数据
dataset = pd.read_csv('./data/1_oxidation_ml_dataset.csv')