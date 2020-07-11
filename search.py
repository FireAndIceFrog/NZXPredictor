import  pandas_datareader as web
import numpy as np
import pandas as  pd 
from datetime import datetime
# Get given input parameters, search and then save the search as *search.csv*
class searchNode:
    def __init__(self,bank = 0, shares = 0,path = "",heuristic = 0,index = 0):
        self.bank = bank
        self.shares = shares
        self.heuristic = heuristic
        self.path = path
        self.index = index
    def setBank(self,num):
        self.bank = num
        return self
    def setShares(self,num):
        self.shares = num
    def setHeuristic(self,num):
        self.heuristic = num
        return self
    def getHeuristic(self,currentPrice):
        return self.bank + self.shares*currentPrice

    def buy(self,price):
        # Buy with all the money in the bank, transfer it to the shares
        if(self.bank == 0): return None
        child = searchNode(0, self.bank / price,self.path+"B",heuristic = (self.bank), index = self.index +1)
        return child

    def sell(self,price):
        # Sell all shares and move to the bank
        if(self.shares == 0): return None
        child = searchNode(self.shares * price, 0, self.path+"S",heuristic = (self.shares * price),index = self.index +1)
        return child
    def hold(self):
        # No change
        child = searchNode(self.bank, self.shares, self.path+"W",heuristic = (self.heuristic),index = self.index +1)
        return child
    def __lt__(self, other):
         return (self.heuristic+self.index) > (other.heuristic+self.index)
    
class SearchTree:
    def __init__(self,pricesList):
        self.searchQueue = []
        self.pricesList = pricesList
        self.priceLength = len(pricesList)
        self.path = ""
        self.finalEarnings = 0
    def setup(self,starting = 50):
        # Set up the search tree with one variable in it
        self.searchQueue = [searchNode().setBank(starting).setHeuristic(starting)]

    def search(self,index=0):
        item = self.searchQueue[0]
        while(item.index != self.priceLength):
            # Begin by reading the search queue
            item = self.searchQueue[0]
            # print("Selected: ",item.heuristic)
            # Remove the 0th item from the queue
            self.searchQueue.pop(0)
            # test for final position

            if (self.priceLength == item.index): 
                break
            # Test buy, sell, ignore
            buy = item.buy(self.pricesList[item.index])
            sell = item.sell(self.pricesList[item.index])
            hold = item.hold()
            # Append to list
            
            if (buy != None) : self.searchQueue.append(buy) 
            if (sell != None) :self.searchQueue.append(sell)
            self.searchQueue.append(hold)
            # Sort the queue by heuristic
            self.searchQueue.sort()
            # print("New Heuristic:")
            # for node in self.searchQueue:
            #     print("\t",node.heuristic)
        self.path = item.path
        self.finalEarnings = item.bank+item.shares*self.pricesList[-1]
        return item.path
    def printResults(self,starting = 50):
        print("Path: ",self.path)
        print("Final earnings: ",self.finalEarnings)
        print("From $",starting,", You would have made ",100*(self.finalEarnings/starting), "% profit")
        print("----------------")
    def writeToFile(self,file):
        f = open(file, "w")
        # W -> 1 0 0
        # S -> 0 1 0
        # B -> 0 0 1
        f.write("Wait,Sell,Buy\n")
        for char in self.path:
            if char == "W":
                f.write("1,0,0\n")
            elif char == "S":
                f.write("0,1,0\n")
            elif char == "B":
                f.write("0,0,1\n")

        f.close()


def search(validationVal):
    df=web.DataReader(validationVal+".nz", data_source = 'yahoo', start = '2012-01-01', end = datetime.today().strftime('%Y-%m-%d'))
    # Get the prices list
    pricesList = df['Close'].values
    tree = SearchTree(pricesList)
    tree.setup()
    tree.search()
    print("Results for: ", validationVal)
    tree.printResults(50)
    tree.writeToFile(validationVal+".csv")

if __name__ == "__main__":
    
    validationVals = ["ABA","AFC","AFI","AFT","AGG","AIA","AIR","ALF","AMP","ANZ","AOR","APA","APL","ARB","ARG","ARV","ASD","ASF","ASP","ASR","ATM","AUG","AWF","BFG","BGI","BGP","BIT","BLT","BOT","BRM","CAV","CBD","CDI","CEN","CGF","CMO","CNU","CO2","CRP","CVT","DGL","DIV","DOW","EBO","EMF","EMG","ENS","ERD","ESG","EUF","EUG","EVO","FBU","FCT","FNZ","FPH","FRE","FSF","FWL","GBF","GEN","GENWB","GEO","GFL","GMT","GNE","GSH","GTK","GXH","HFL","HGH","HLG","IFT","IKE","IPL","JLG","JPG","JPN","KFL","KFLWF","KMD","KPG","LIC","LIV","MCK","MCKPA","MCY","MDZ","MEE","MEL","MET","MFT","MGL","MHJ","MLN","MLNWD","MMH","MOA","MPG","MWE","MZY","NPF","NPH","NTL","NTLOB","NWF","NZB","NZC","NZK","NZM","NZO","NZR","NZX","OCA","OZY","PCT","PCTHA","PEB","PFI","PGW","PIL","PLP","PLX","POT","PPH","PYS","QEX","RAK","RBD","RYM","SAN","SCL","SCT","SCY","SDL","SEA","SEK","SKC","SKL","SKO","SKT","SML","SNC","SNK","SPG","SPK","SPN","SPY","SRF","STU","SUM","TCL","TEM","TGG","THL","TLL","TLS","TLT","TNZ","TPW","TRA","TRS","TRU","TWF","TWR","USA","USF","USG","USM","USS","USV","VCT","VGL","VHP","VTL","WBC","WDT","WHS","ZEL"]

    # for validationVal in validationVals:
    #     search(validationVal)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(search, validationVal): validationVal for validationVal in validationVals}
        for future in concurrent.futures.as_completed(future_to_url):
            
            data = future.result()
    
