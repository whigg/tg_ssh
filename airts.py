import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from datetime import datetime
#rcParams设定好画布的大小
rcParams['figure.figsize'] = 15, 6

# read data
data = pd.read_csv("./data/AirPassengers.csv")
print(data.head())
print('\n Data types:')
print(data.dtypes)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
# 其中parse_dates 表明选择数据中的哪个column作为date-time信息，
# index_col 告诉pandas以哪个column作为 index
#  date_parser 使用一个function(本文用lambda表达式代替)，使一个string转换为一个datetime变量
data = pd.read_csv('./data/AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
print(data.head())
print(data.index)

