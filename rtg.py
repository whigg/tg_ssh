# coding=UTF-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import tide data using pandas
tg_san_usa = pd.read_csv('./data/10.rlrdata', sep=';', encoding='utf-8')
#

print(type(tg_san_usa.values[:, 0]))
print(type(tg_san_usa))

temp = tg_san_usa.values[:, 1]  # ssh values
tim = tg_san_usa.values[:, 0]   # time values
idx = temp != -99999  # NOT 无效值
temp = temp[idx]
tim = tim[idx]
print(len(temp))
print(len(tg_san_usa.iloc[:, 1]))
# 绘制TG时间序列
plt.plot(tim, temp)
plt.show()

# 趋势分析
# SVR机器学习算法
# Bayes
