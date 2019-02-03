import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt

# 读取数据，pd.read_csv默认生成DataFrame对象，需将其转换成Series对象
df = pd.read_csv('./data/AirPassengers.csv', encoding='utf-8', index_col='date')
df.index = pd.to_datetime(df.index)  # 将字符串索引转换成时间索引
ts = df['x']  # 生成pd.Series对象
# 查看数据格式
print(ts.head())
print(ts.head().index)
print(ts['1949-01-01'])
print(ts[datetime(1949, 1, 1)])
print(ts['1949'])
print(ts['1949-1': '1949-6'])

ts_log = np.log(ts)
test_stationarity.draw_ts(ts_log)