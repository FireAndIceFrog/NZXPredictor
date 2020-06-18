
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.models import model_from_json
import matplotlib.pyplot as plt

plt.style.use ('fivethirtyeight')


df = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end = '2019-12-17')
# print(df)

plt.figure(figsize = (16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
# plt.show()

#Create a new DF With only the close column
data = df.filter(['Close'])
#convert the datafram to a numpy arr
dataset = data.values
# Get the number of rows to train the model
training_data_len = math.ceil(len(dataset) *0.8)
print("Data length: ",len(dataset),"\nTraining Length: ",training_data_len)

# scale the data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)

# print(scaled_data)

# create the training dataset
# Create the scaed training set
train_data = scaled_data[0:training_data_len, :]
# Split the data into X_train and Y_train dataset
# Independant variables
x_train = []
# target
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])
    

# Convert to numpy arrs
x_train,y_train = np.array(x_train), np.array(y_train)

#reshape the data
rows, cols = x_train.shape
x_train = np.reshape(x_train, (rows,cols,1))


#build the LSTM Model
model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape = (cols,1)))
model.add(LSTM(50,return_sequences = False ))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# train
model.fit(x_train, y_train,batch_size = 1, epochs = 1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


#create the testing data set
# create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len-60:, :]
# create the datasets X-test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range( 60, len(test_data)):
    x_test.append(test_data[i-60:i,0])
# convert the data to a numpy array
x_test = np.array(x_test)
# reshape the data 
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(predictions-y_test)**2)

print(rmse)


# plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USE ($)',fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid['Close', 'Predictions'])
plt.legend(['Train','Val', 'Predictions'], loc = 'lower right')
plt.show()




























