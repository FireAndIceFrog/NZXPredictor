
#this converter will be the base class for all functors; it will be created so that there is a reference point for all the predictors
# This will have 3 command:
# Predict():
#  -Called to convert a dataframe of trades (containing high, low, open, close, volume)
#  -Stores the DF in memory
#  -This function will reduce the DF to one data stream, normalized between -1 and 1
# setDF(DF)
#  -Called to set up the DF, and checks that it has the correct rows to use
# getNP()
#  -returns the numpy array of the DF from the file.
class NotEnoughCols(Exception):
    """The required DF needs to have the columns { 'Volume', 'Open','Close','High','Low' }"""
    pass
class ArrayNotInitialized(Exception):
    """You must call predict before you call getNP!"""
    pass
class Null(Exception):
    """You not enough input parameters - did you call setDF?"""
    pass
class converter:
    def __init__(self):
        self.df = None
        self.returnable = None
    def setDF(self, df):
        if not {'Volume', 'Open','Close','High','Low'}.issubset(df.columns):
            raise NotEnoughCols
        self.df = df
    def getDF(self):
        try:
            if(self.returnable == None):
                raise ArrayNotInitialized
        except:
            
            return self.returnable
            
    def predict(self):
        try:
            if self.df == None:
                raise Null
        except: 
            pass

    
