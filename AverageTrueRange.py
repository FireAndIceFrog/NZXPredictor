import abstractConverter as AC 
import pandas as  pd
import numpy as np
from math import exp
import stockstats
#Average true range is an indicator which tells us how strong the trend is strong or weak
# We will compare ADX to ADXR to generate buy/sell signals.
# This will output between -1 and 1


class ADX(AC.converter):
    def __init__(self,key,scaler = 6):
        super().__init__()
        # used for accessing the data in the new DF
        self.key = key
        self.scaler = scaler
    def setScalar(self,scaler=6):
        self.scaler = scaler
    
    def setDF(self, df):
        self.df = stockstats.StockDataFrame.retype(df)
    def predict(self):
        super().predict()
        self.returnable = pd.DataFrame([], columns = [ self.key])
        self.returnable[self.key] = self.df.loc[:,'adx'].sub(self.df.loc[:,'adxr']).apply(lambda num: (1/(1+exp(-num/self.scaler))))
        

