import abstractConverter as AC 
import pandas as  pd
import numpy as np
from math import exp

#Moving average convergence divergence is the idea that you buy when the shorter MA (the one with a smaller window) is above the larger MA, and sell when visa versa
# We want to have a very select set of cases, the system needs to output 1 when its above the long MA, and -1 when its below
# We will use a modified sigmoid functor:
#           2            
#       _________________ + 1
#       1+e^(-difference/scalar)
# This will output a result between 0 and 1
# 


class MACD(AC.converter):
    def __init__(self,key,shortMA = 1, LongMA = 2):
        super().__init__()
        # used for accessing the data in the new DF
        self.key = key
        self.scaler = 1
        # Make sure that the short MA is always the short MA
        if(shortMA <= LongMA):
            self.shortMA = shortMA
            self.longMA = LongMA
        else :
            self.shortMA = LongMA
            self.longMA = shortMA
    def setScalar(self,scaler):
        self.scaler = scaler
    def predict(self):
        super().predict()
        self.returnable = pd.DataFrame([], columns = [ self.key])
        shortEMA = pd.DataFrame([], columns = [ 'short'])
        longEMA = pd.DataFrame([], columns = [ 'long'])
        shortEMA[ 'short'] = self.df['Close'].ewm(span = self.shortMA).mean()
        longEMA  [ 'long'] = self.df['Close'].ewm(span = self.longMA).mean()
        # get the returnable
        self.returnable[self.key]= shortEMA.loc[:,'short'].sub(longEMA.loc[:,'long']).apply(lambda num: (1/(1+exp(-num/self.scaler))))


