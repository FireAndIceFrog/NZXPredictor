
# import math
# import pandas_datareader as web
import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense,LSTM
# from keras.models import model_from_json
# import matplotlib.pyplot as plt

# plt.style.use ('fivethirtyeight')


# df = web.DataReader('ABA.NZ', data_source = 'yahoo', start = '2012-01-01', end = '2019-12-17')
# print(df)

# plt.figure(figsize = (16,8))
# plt.title('Close price history')
# plt.plot(df['High'])
# plt.xlabel('Date', fontsize = 18)
# plt.ylabel('Close Price USD ($)', fontsize = 18)
# plt.show()







listData = np.array([[1,2,3],[4,5,6],[7,8,9]])
print (listData)

inputData = \
[
    [[0.1,0.1]],
    [[0.1,0.9]],
    [[0.9,0.1]],
    [[0.9,0.9]]
]


test = np.array(inputData)

print(test)
print("List data shape: ", test.shape)













