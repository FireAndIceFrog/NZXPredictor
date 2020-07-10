
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM, Conv2D,MaxPool2D,Flatten,Dropout
from keras.models import model_from_json
import matplotlib.pyplot as plt
from datetime import datetime
from Main import getTargetBuySell,getCombinedDF


plt.style.use ('fivethirtyeight')
strings = ["ABA","AFC","AFI","AFT","AGG","AIA","AIR","ALF","AMP","ANZ","AOR","APA","APL","ARB","ARG","ARV","ASD","ASF","ASP","ASR","ATM","AUG","AWF","BFG","BGI","BGP","BIT","BLT","BOT","BRM","CAV","CBD","CDI","CEN","CGF","CMO","CNU","CO2","CRP","CVT","DGL","DIV","DOW","EBO","EMF","EMG","ENS","ERD","ESG","EUF","EUG","EVO","FBU","FCT","FNZ","FPH","FRE","FSF","FWL","GBF","GEN","GENWB","GEO","GFL","GMT","GNE","GSH","GTK","GXH","HFL","HGH","HLG","IFT","IKE","IPL","JLG","JPG","JPN","KFL","KFLWF","KMD","KPG","LIC","LIV","MCK","MCKPA","MCY","MDZ","MEE","MEL","MET","MFT","MGL","MHJ","MLN","MLNWD","MMH","MOA","MPG","MWE","MZY","NPF","NPH","NTL","NTLOB","NWF","NZB","NZC","NZK","NZM","NZO","NZR","NZX","OCA","OZY","PCT","PCTHA","PEB","PFI","PGW","PIL","PLP","PLX","POT","PPH","PYS","QEX","RAK","RBD","RYM","SAN","SCL","SCT","SCY","SDL","SEA","SEK","SKC","SKL","SKO","SKT","SML","SNC","SNK","SPG","SPK","SPN","SPY","SRF","STU","SUM","TCL","TEM","TGG","THL","TLL","TLS","TLT","TNZ","TPW","TRA","TRS","TRU","TWF","TWR","USA","USF","USG","USM","USS","USV","VCT","VGL","VHP","VTL","WBC","WDT","WHS","ZEL"]



model = Sequential()
# model.add(LSTM(100,return_sequences = True, input_shape = (60,5),use_bias = True))
# model.add(LSTM(100,return_sequences = False, use_bias = True))
# model.add(Dense(50, use_bias = True))
# model.add(Dense(25, use_bias = True))
# model.add(Dense(1,use_bias = True))


model.add(Conv2D(32,(16,16),activation = 'relu', input_shape = (1,3,5),padding = 'same',init='glorot_uniform'))
model.add(MaxPool2D((2,2),padding='same'))

#second round
model.add(Conv2D(48,(3,3),activation = 'relu', padding = 'same' ,init='glorot_uniform'))
model.add(MaxPool2D((2,2),padding='same'))

# Fourth round
model.add(Conv2D(64,(3,3),activation = 'relu', padding = 'same',init='glorot_uniform' ))
model.add(MaxPool2D((2,2),padding='same'))

model.add(Conv2D(96,(3,3),activation = 'relu', padding = 'same',init='glorot_uniform' ))
model.add(MaxPool2D((2,2),padding='same'))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(1,use_bias = True))

model.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')

print(model.input_shape)
print(model.output_shape)
for string in strings:
    df = web.DataReader(string+".nz", data_source = 'yahoo', start = '2012-01-01', end = datetime.today().strftime('%Y-%m-%d'))
    #Create a new DF With only the close column
    dataset,keys = getCombinedDF(address = string+".nz")
    targetData = getTargetBuySell(address = string+".nz")
    # data = df.filter(['Close'])
    # print (data)
    #convert the datafram to a numpy arr
    # dataset = data.values
    # Get the number of rows to train the model
    training_data_len = math.ceil(len(dataset) *1)
    print("Data length: ",len(dataset),"\nTraining Length: ",training_data_len)

    # scale the data
    

    # print(scaled_data)

    # create the training dataset
    train_data = dataset[0:training_data_len, :]
    # Split the data into X_train and Y_train dataset
    # Independant variables
    x_train = []
    # target
    y_train = []

    for i in range(3, len(train_data)):
        x_train.append(train_data[i-3:i, :])
        y_train.append(targetData[i,:])
        

    # Convert to numpy arrs
    x_train,y_train = np.array(x_train), np.array(y_train)

    #reshape the data
    rows, cols, order = x_train.shape
    # print(x_train)
    # x_train = np.reshape(x_train, (rows,cols,4))
    #build the LSTM Model
    # batch = []
    # batchTest = []
    # for i in range(100, x_train.shape[0], 100):
    #     batch.append( x_train[i-100:i,:])
    # batch = np.array(batch)

    x_train = np.expand_dims(x_train,axis=1)
    print(x_train.shape)

    
    # train
    model.fit(x_train, y_train,batch_size = 1, epochs = 10)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")