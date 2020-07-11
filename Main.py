from movingAverage import movingAverage as MA
import numpy as np
import pandas as  pd 
from datetime import datetime
import  pandas_datareader as web
import matplotlib.pyplot as plt
import matplotlib
from macd import MACD
from AverageTrueRange import ADX
from rsi import RSI
from evolution import *
from os import system
from  ProgressBar import printProgressBar
def getResult(sigClass,df,scaler):
    # Input the data frame
    sigClass.setDF( df)
    # Set the scaler(Could be any float)
    sigClass.setScalar(scaler)
    # Get the distance from the moving average
    sigClass.predict()
    # Return the MA
    return sigClass.getDF()
def getMAsig(df, window, scaler, key = "mainMA"):
    # Set up the moving average
    Average = MA(window,key)
    return getResult(Average,df,scaler)

def getMACDSig(df, short, long, scaler, key = "mainMA"):
    # Set up the moving average
    Average = MACD(key, short, long)
    return getResult(Average,df,scaler)

def getADXsig(df, key,scaler = 6):
    # Set up the Average true range; this generates buy/sell signals based on how STRONG a trend is 
    Average = ADX(key, scaler)
    return getResult(Average,df,scaler)

def getRSIsig(df,key,scaler = 1):
    # Set up the moving average
    Average = RSI(key, scaler)
    return getResult(Average,df,scaler)

def plot(key, df, show = True, labelx = 'Distributions:'):
    df[key].plot(label=labelx)

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')
    if(show):
        plt.show()



def getCombinedDF(minMA = 50,address = 'aba.nz'):
    keys = []
    key = "mainMA"
    df=web.DataReader(address, data_source = 'yahoo', start = '2012-01-01', end = datetime.today().strftime('%Y-%m-%d'))
    MADist = getMAsig(df, minMA, 1, key)
    plot(key, MADist, show = False, labelx = 'MA')
    keys.append(key)    
    key = "MACD"
    MACDsig = getMACDSig(df, 6,9,0.08,key)
    plot(key,MACDsig,show = False,labelx = 'MACD')
    keys.append(key)
    key = "ADX"
    ADXsig = getADXsig(df,key)
    plot(key,ADXsig,show = False, labelx = 'ADX')
    keys.append(key)
    key = "RSI"
    RSIsig = getRSIsig(df,key)
    plot(key,RSIsig,show = False, labelx = 'RSI')
    keys.append(key)
    # Reset the DF
    df=web.DataReader(address, data_source = 'yahoo', start = '2012-01-01', end = datetime.today().strftime('%Y-%m-%d'))
    sigDF = MADist
    sigDF["MACD"] = MACDsig
    sigDF["ADX"]= ADXsig
    sigDF["RSI"] = RSIsig 
    sigDF["High"] = df['High']
    sigDF["Low"] = df['Low']
    sigDF["Volume"] = df['Volume']
    sigDF["Open"] = df['Open']
    # Remove invalid data points
    # Remember: Remove the initial values from the DF. if you have NAN numbers it will damage the AI
    sigDF = sigDF[minMA:]
    # print(sigDF)
    return (sigDF.to_numpy(),keys)

class sigGen:
    def __init__(self, df):
        self.df = df
        self.bool = True
    def onclick(self,event):
        if(self.bool):
            self.bool = False
            self.df.loc[matplotlib.dates.num2date(event.xdata)] = 1
        else:
            self.bool = True
            self.df.loc[matplotlib.dates.num2date(event.xdata)] = -1
        print('double' if event.dblclick else 'single', event.button,
            event.x, event.y, matplotlib.dates.num2date(event.xdata), event.ydata)
    def getDF(self):
        return self.df

if __name__ == "__main__":
    minMA = 50
    data=web.DataReader('aba.nz', data_source = 'yahoo', start = '2012-01-01', end = datetime.today().strftime('%Y-%m-%d'))["Close"][minMA:]
    template = data.apply(lambda a : 0)
    close = data.to_numpy()

    sigDF,keys = getCombinedDF(minMA)
    start = 1
    end = 100
    numGenerations = 50
    algo = evolver(num_inputs = len(keys) ,num_hidden = 100,pop_size = 100, template = template)
    printProgressBar(0, numGenerations, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for j in range(numGenerations):
        for i in range(len(sigDF)-1):
            algo.propogate(sigDF[i],close[i],i)
        algo.calcFitness(close[-1])
        if(algo.population[0].bank+algo.population[0].shares*close[-1] > 2000):
            break
        # results = [(algo.population[x].bank+algo.population[x].shares*close[-1]) for x in range(len(algo.population))]
        results = algo.population[0].bank+algo.population[0].shares*close[i]
        transacs = algo.population[0].transactions
        
        algo.repopulate()
        # system("cls")
        
        
        printProgressBar(j + 1, numGenerations, prefix = 'Progress:', suffix = 'Complete', length = 50)  
        print("\nTop Earner: ",results)
        print("Transactions: ",transacs)
            
            

        
        
    # Plot the result of a generation
    fig=plt.figure()
    ax=fig.add_subplot(111)
    data.plot(label="close")
    ax.plot(algo.population[0].df.loc[algo.population[0].df==1].index,data[ algo.population[0].df.loc[algo.population[0].df==1].index],label='BUY',lw=0,marker='^',c='g')
    ax.plot(algo.population[0].df.loc[algo.population[0].df==-1].index,data[ algo.population[0].df.loc[algo.population[0].df==-1].index],label='SELL',lw=0,marker='v',c='r')
    # ax.plot(algo.population[0].df==-1,label='SHORT',lw=0,marker='v',c='r')
    plt.show()

   