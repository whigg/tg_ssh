# coding=UTF-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define function

# Import tide data using pandas
# set names of column
tg_san_usa = pd.read_csv('./data/10.rlrdata', sep=';', encoding='utf-8', header=None, names=['years', 'height', 'tag1',
                                                                                             'tag2'])
# 读文件的时候，如果文件没有列名称，可以自己定义。

# Test pandas read function
# print(type(tg_san_usa.values[:, 0]))
# print(type(tg_san_usa))
print("Quick look at the tg data: \n", tg_san_usa.head())

# Define time
temp = tg_san_usa['height']
tim = tg_san_usa['years']
# temp = tg_san_usa.values[:, 1]  # ssh values
# tim = tg_san_usa.values[:, 0]   # time values

# 判断数据中的无效值
idx = temp != -99999  # NOT 无效值
tg_data = temp[idx]
tim_data = tim[idx]
print("The tg data length without -9999:", len(tg_data))
print("The tg data length with -9999:", len(tg_san_usa.iloc[:, 1]))

# 提取数据无效值，并保存到指定数组,在后续的统计分析加入无效值的权重。Bayes分析？
idx = temp == -99999  # NOT 无效值
tg_non = temp[idx]
tim_non = tim[idx]
print("The NaN (-9999) data length:", len(tg_non), ". And the time of NaN data:", tim_non)

# 绘制TG时间序列
plt.plot(tim_data, tg_data, color='blue', linewidth=0.5, linestyle='-')
plt.xlabel('Year')
plt.ylabel('tg ssh/mm')
plt.show()

# 趋势分析,使用简单的方法计算trend


# SVR机器学习算法

# Bayes
# Model refine. Add the uncertainties of the TG data. Add the land vertical motion.
