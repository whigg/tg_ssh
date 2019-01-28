# coding=UTF-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tg_san_usa = pd.read_csv('10.rlrdata', sep=';', encoding='utf-8')

temp = np.asarray(tg_san_usa.iloc[:, 1])
tim = np.asarray(tg_san_usa.iloc[:, 0])
idx = temp != -99999  # 无效值
temp = temp[idx]
tim = tim[idx]

# 绘制TG时间序列
plt.plot(tim, temp)
plt.show()

# 趋势分析
# SVR机器学习算法