
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.models import model_from_json
import matplotlib.pyplot as plt
from datetime import datetime
from os import system, name 

plt.style.use ('fivethirtyeight')
inp = "asdg"

while(inp != "q"):
    _ = system('cls') 
    print("Enter code: ")
    inp = input()
    
    df = web.DataReader(inp, data_source = 'yahoo', start = '2012-01-01', end = datetime.today().strftime('%Y-%m-%d'))
    # print(df)


    #Create a new DF With only the close column
    data = df.filter(['Close','High','Low','Open','Volume'])
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
        x_train.append(train_data[i-60:i, :])
        y_train.append(train_data[i,0])
        

    # Convert to numpy arrs
    x_train,y_train = np.array(x_train), np.array(y_train)

    #reshape the data
    rows, cols, order = x_train.shape
    # x_train = np.reshape(x_train, (rows,cols,1))


    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    model.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')



    #create the testing data set
    # create a new array containing scaled values from index 1543 to 2003
    test_data = scaled_data[training_data_len-60:, :]
    # create the datasets X-test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range( 60, len(test_data)):
        x_test.append(test_data[i-60:i,:])
    # convert the data to a numpy array
    # _ = system('cls') 
    x_test = np.array(x_test)
    print(x_test)
    print(y_test)
    x_test = np.expand_dims(x_test,axis=1)
    # reshape the data 
    # x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    rows = predictions.shape[0]
    z = np.zeros((rows,4))
    predictions = np.append(predictions,z,axis=1)
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions[:,0]

    # get the root mean squared error(RMSE)
    rmse = np.sqrt(np.mean(predictions-y_test[:,0])**2)
    
    print("Results: ")
    print("Root Mean Squared Error", rmse)


    # plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # print(valid)
    # print(predictions)
    # valid['Predictions'] = predictions
    plt.figure(figsize=(16,8))
    plt.title("Model")
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price USE ($)',fontsize = 18)
    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.legend(['Train','Val', 'Predictions'], loc = 'lower right')
    plt.show()
    




























