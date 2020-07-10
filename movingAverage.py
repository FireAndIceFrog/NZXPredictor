import abstractConverter as AC 
import pandas as  pd
import numpy as np

#moving average class, this will use fuzzy logic to determine how close to the moving average it is.
# When the price is close to the moving average, output a 0
# When the price is above the moving average, output a 1
# when the price is below the moving average, output a -1
# The output will be a parabolic curve where calculatedPrice = min(max(scaler*(price-moving average)^2,-1)1)
#     -------------------------(1)
#    /
#   /
#  /
# /
# |----------moving average 0
# \
#  \ 
#   \ 
#    \
#     \
#      ------------------------(-1)
# 

class movingAverage(AC.converter):
    def __init__(self,movingAverageRange, key):
        super().__init__()
        self.movingAverage = movingAverageRange
        # used for accessing the data in the new DF
        self.key = key
        self.scaler = 1
    def setScalar(self,scaler):
        self.scaler = scaler
    def predict(self):
        super().predict()
        signals=pd.DataFrame([], columns = [ 'MainMovingAverage'])
        self.returnable = pd.DataFrame([], columns = [ self.key])
        # get the Simple moving average
        signals['MainMovingAverage']=self.df['Close'].rolling(window=self.movingAverage,min_periods=1,center=False).mean()
        # Return the distance from the main moving average
        # We divide by absolute value so that we keep the original sign (-1, 1)
        self.returnable[self.key]= self.df.loc[:,'Close'].sub(signals.loc[:,'MainMovingAverage']).apply(lambda num: min(1.0,max(-1.0, num if (num == 0) else num/abs(num)*self.scaler*(num)**2)))

