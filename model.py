
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM, Conv1D,MaxPooling1D,Flatten,Dropout
from keras.models import model_from_json
import matplotlib.pyplot as plt
from datetime import datetime
from Main import getCombinedDF


plt.style.use ('fivethirtyeight')
# strings = ["ABA","AFC","AFI","AFT","AGG","AIA","AIR","ALF","AMP","ANZ","AOR","APA","APL","ARB","ARG","ARV","ASD","ASF","ASP","ASR","ATM","AUG","AWF","BFG","BGI","BGP","BIT","BLT","BOT","BRM","CAV","CBD","CDI","CEN","CGF","CMO","CNU","CO2","CRP","CVT","DGL","DIV","DOW","EBO","EMF","EMG","ENS","ERD","ESG","EUF","EUG","EVO","FBU","FCT","FNZ","FPH","FRE","FSF","FWL","GBF","GEN","GENWB","GEO","GFL","GMT","GNE","GSH","GTK","GXH","HFL","HGH","HLG","IFT","IKE","IPL","JLG","JPG","JPN","KFL","KFLWF","KMD","KPG","LIC","LIV","MCK","MCKPA","MCY","MDZ","MEE","MEL","MET","MFT","MGL","MHJ","MLN","MLNWD","MMH","MOA","MPG","MWE","MZY","NPF","NPH","NTL","NTLOB","NWF","NZB","NZC","NZK","NZM","NZO","NZR","NZX","OCA","OZY","PCT","PCTHA","PEB","PFI","PGW","PIL","PLP","PLX","POT","PPH","PYS","QEX","RAK","RBD","RYM","SAN","SCL","SCT","SCY","SDL","SEA","SEK","SKC","SKL","SKO","SKT","SML","SNC","SNK","SPG","SPK","SPN","SPY","SRF","STU","SUM","TCL","TEM","TGG","THL","TLL","TLS","TLT","TNZ","TPW","TRA","TRS","TRU","TWF","TWR","USA","USF","USG","USM","USS","USV","VCT","VGL","VHP","VTL","WBC","WDT","WHS","ZEL"]

strings = ["AGG","aba","aia","air","AMP","ANZ","kmd"]

model = Sequential()

model.add(Conv1D(filters=128, kernel_size=3, input_shape=(60,4),use_bias=True))
model.add(MaxPooling1D(pool_size=2))

#second round
model.add(Conv1D(filters=128, kernel_size=3,use_bias=True))
model.add(MaxPooling1D(pool_size=2))

# Fourth round
model.add(Conv1D(filters=64, kernel_size=3,use_bias=True))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=3,use_bias=True))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(3,use_bias = True,activation = 'sigmoid'))

model.compile(optimizer = 'SGD', loss = 'mean_squared_error')

print(model.input_shape)
print(model.output_shape)
for string in strings:
    df = web.DataReader(string+".nz", data_source = 'yahoo', start = '2012-01-01', end = datetime.today().strftime('%Y-%m-%d'))
    #Create a new DF With only the close column
    # dataset,keys = getCombinedDF(address = string+".nz")
    dataset = df[["High","Low","Open","Volume"]].values
    
    # Scale the data from 0 -> 1
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    # dataset = np.array(df[['Open', 'High','Low','Volume']])
    targetData = pd.read_csv(string+".csv")
    
    # data = df.filter(['Close'])
    # print (data)
    #convert the datafram to a numpy arr
    # dataset = data.values
    # Get the number of rows to train the model
    training_data_len = math.ceil(len(dataset) *0.8)
    print("Data length: ",len(dataset),"\nTraining Length: ",training_data_len)

    # Convert Target to NP,removing the data entry for the first 50 (as they are cut off)
    targetData = np.array(targetData[50:])
    print("Both datasets are of size: ",len(targetData)," ",len(dataset))
    # create the training dataset
    slidingData = []
    slidingRange = 60
    for i in range(slidingRange,training_data_len):
        slidingData.append(dataset[i-slidingRange:i,:])
    slidingData = np.array(slidingData)
    print(slidingData.shape)

    
    # remove irrelevant data from the target
    x_train = slidingData
    targetData = targetData[slidingRange:training_data_len,:]

    slidingData = []
    slidingRange = 60
    for i in range(training_data_len+slidingRange,len(dataset)):
        slidingData.append(dataset[i-slidingRange:i,:])
    slidingData = np.array(slidingData)
    print(slidingData.shape)

    x_test = slidingData
    TrainData = targetData[training_data_len+slidingRange:,:]

    print("Both datasets are of size: ",len(x_train)," ",len(targetData))
        

    # Convert to numpy arrs
    x_train,x_test = np.array(x_train),np.array(x_test)

    #reshape the data
    # print(x_train)
    # x_train = np.reshape(x_train, (rows,cols,4))
    #build the LSTM Model
    # batch = []
    # batchTest = []
    # for i in range(100, x_train.shape[0], 100):
    #     batch.append( x_train[i-100:i,:])
    # batch = np.array(batch)

    print("Train shape: ",x_train.shape)
    print("Model input: ",model.input_shape)

    # # train
    model.fit(x_train, targetData,batch_size = 1, epochs = 40)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")