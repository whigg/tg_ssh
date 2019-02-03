import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm

# Subsetting the dataset
# Index 11856 marks the end of year 2013
df = pd.read_csv('./data/train.csv', nrows=11856)

#  Creating train and test set
# Index 10392 marks the end of October 2013
train = df[0:10392]
test = df[10392:]

# Aggregating the dataset at daily level
# now = pd.Timestamp.now()
# print(now)  # Get the time now
# now_sh = now.tz_localize("Asia/Shanghai")
# print(pd.Timestamp(2017, 1, 1, 12))

df['Timestamp'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
df.index = df['Timestamp']
df = df.resample('D').mean()  # D表示day,这里是日均值的重采样，前面设置了日期的格式，这里可以使用这种表达,
# 还可以用0.5D,M,Y,H等时间格式

train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
train.index = train['Timestamp']
train = train.resample('D').mean()

test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%d-%m-%Y %H:%M')
test.index = test['Timestamp']
test = test.resample('D').mean()

# Plotting data
train.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
test.Count.plot(figsize=(15, 8), fontsize=14)
# plt.show()
#
dd = np.asarray(train['Count'])  # 数据，人数
y_hat = test.copy()

y_hat['naive'] = dd[len(dd) - 1]
plt.figure(figsize=(12, 8))
plt.plot(train.index, train['Count'], label='Train')
plt.plot(test.index, test['Count'], label='Test')
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
#
# plt.show()
#
rms = sqrt(mean_squared_error(test['Count'], y_hat['naive']))
print(rms)
sm.tsa.seasonal_decompose(train['Count']).plot()
result = sm.tsa.stattools.adfuller(train['Count'])
# plt.show()
#
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()
