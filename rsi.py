import abstractConverter as AC 
import pandas as  pd
import numpy as np
from math import exp
import stockstats
#Moving average convergence divergence is the idea that you buy when the shorter MA (the one with a smaller window) is above the larger MA, and sell when visa versa
# We want to have a very select set of cases, the system needs to output 1 when its above the long MA, and -1 when its below
# We will use a modified sigmoid functor:
#           2            
#       _________________ + 1
#       1+e^(-difference/scalar)
# This will output a result between 0 and 1
# 


class RSI(AC.converter):
    def __init__(self,key,scaler = 1):
        super().__init__()
        # used for accessing the data in the new DF
        self.key = key
        self.scaler = scaler
    def setScalar(self,scaler=1):
        self.scaler = scaler
    
    def setDF(self, df):
        self.df = stockstats.StockDataFrame.retype(df)
    def predict(self):
        super().predict()
        self.returnable = pd.DataFrame([], columns = [ self.key])
        # We want 1 to be the good signal and -1 to be the bad signal.
        # For RSI, a value over 70 is considered BAD
        # To fix this, we are going to times the value by the negation plus 100
        self.returnable[self.key] = self.df.loc[:,'rsi_12'].apply(lambda a : (a)/100.0)

if __name__ == "__main__":
    
    from datetime import datetime
    import  pandas_datareader as web
    import matplotlib.pyplot as plt
    df=web.DataReader('aba.nz', data_source = 'yahoo', start = '2012-01-01', end = datetime.today().strftime('%Y-%m-%d'))
    key = "as"
    RSIsig = RSI(key, 1)
    RSIsig.setDF(df)
    RSIsig.predict()
    sig = RSIsig.getDF()
    print(sig)
    sig.plot(label="sig")
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')
    plt.show()