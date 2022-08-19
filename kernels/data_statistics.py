# -*- coding:utf-8 -*-

import pandas as pd

train_file = "../data/训练集/train.csv"
test_file = "../data/测试集/test1.csv"

train = pd.read_csv(train_file, sep="\t")
test = pd.read_csv(test_file, sep="\t")
print(train[:10], len(train))  # 45249 条
print(test[:10], len(test))    # 1040 条

# 训练集数据总类别标签分布统计
print(train['label'].value_counts())  # 1:33760, 0:11488
# 1    33760
# 0    11488
# Name: label, dtype: int64

# 训练集文本长度统计分析, 分析可得文本属中长文本，最长231字
print(train['text'].map(len).describe())
# count    45248.000000
# mean        50.425544
# std         16.476572
# min          8.000000
# 25%         38.000000
# 50%         52.000000
# 75%         62.000000
# max        231.000000
# Name: text, dtype: float64

# 测试集文本长度统计分析
print(test["text"].map(len).describe())
# count    1039.000000
# mean       27.879692
# std         9.289082
# min         9.000000
# 25%        21.000000
# 50%        26.000000
# 75%        33.000000
# max        74.000000
# Name: text, dtype: float64

