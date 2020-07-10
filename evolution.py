import numpy as np 
import pandas as pd 
import math 
from concurrent.futures import ThreadPoolExecutor
from random import random
from numba import jit

def createLayers(layers = 3,
num_weights = 4):
    weightsArr = []
    for i in range(layers):
        weightsArr.append(np.random.uniform(low=-1.0, high=1.0, size=num_weights))
    weightsArr = np.array(weightsArr)
    # print(weightsArr)
    return weightsArr



@jit
def matrixMult(A,B,numA,numB):
    outArr = []
    for i in range(numB):
        ABsum = 0.0
        for j in range(numA):
            ABsum = ABsum + A[j]*B[i]
        outArr.append(ABsum)
    return np.array(outArr)

@jit
def activation(num):
    return (2/(1+math.exp(-num)))-1


@jit
def propogate (inputs, weights, layers,num_weights, numInputs):
    # print("Layer,weights: ",layers," ",num_weights)
    output = inputs
    for layer in range(layers-1):
        outputArr = []
        for i in range(num_weights):
            weight = matrixMult(output,weights[layer],numInputs,num_weights)
            weightSum = 0.0
            for j in range(num_weights):
                weightSum = weightSum + weight[j]

            weightSum = activation(weightSum)
            outputArr.append(weightSum)
        output =  np.array(outputArr)
    # Compile to one output
    outputArr = []
    for i in range(num_weights):
        weight = matrixMult(output,weights[-1],numInputs,num_weights)
        weightSum = 0.0
        for j in range(num_weights):
            weightSum = weightSum + weight[j]

        weightSum = activation(weightSum)
        outputArr.append(weightSum)
    weightSum = 0.0
    for j in range(num_weights):
        weightSum = weightSum + outputArr[j]

    weightSum = activation(weightSum)

    output =  np.array(weightSum)
    return output


@jit
def mate(weightsA,weightsB,layers,num_weights):
    mutation = 0.1
    mutationChange = 0.5
    # swap weights between each parent for each layer and times it by the mutation
    child = weightsA
    for layer in range(layers):
        for i in range(num_weights):
            # Choose between parent A and parent B
            if(random() > 0.5):
                child[layer][i] = weightsB[layer][i]
            # Can ignore parent A as it already is there
            # Mutate child
            mutationAmnt = mutation*(random()*2)
            if(mutationAmnt > mutationChange or mutationAmnt<-mutationChange):
                mutate = child[layer][i] * mutationAmnt
                child[layer][i] = child[layer][i]+mutate
    return child




class node:
    def __init__(self, layers, weights,df, generateWeights = True):
        self.template = df
        self.df = df
        self.shares = 0.0
        self.bank = 50.0
        self.layers = layers
        self.weights =weights
        self.brokerage = 0.000
        self.transactions = 0
        self.resetFitness()
        self.lastTrans = False
        self.weightsArr = None
        if(generateWeights):
            self.weightsArr = createLayers(layers = layers,num_weights = weights)

    def mate(self,other):
        child = node(self.layers, self.weights, self.template, generateWeights = False)
        
        child.weightsArr = mate(self.weightsArr, other.weightsArr, self.weightsArr.shape[0],self.weightsArr.shape[1])
        
        return child


    def propogate(self, inputs):
        return propogate(inputs, self.weightsArr,self.weightsArr.shape[0],self.weightsArr.shape[1], inputs.shape[0])
    
    def setBank(self, x):
        self.bank = x
    def buy(self, priceOfShare,DFindex,membership = 1):
        # print("\tBuying with membership: ",membership)
        # Brokerage of %
        availableAmount = self.bank*membership*(1.0+self.brokerage) if self.bank > 1.0   else 0.0
        

        if(availableAmount  <= 0.0): return
        elif(self.bank-availableAmount<=0): return
        self.bank = self.bank-availableAmount
        self.shares = self.shares + (availableAmount/priceOfShare)
        self.transactions = self.transactions +1
        self.df.iloc[DFindex] = 1
        self.lastTrans = True
        # print("Bought: ")
        # print("|\tBANK: ",self.bank)
        # print("|\tShares: ",self.shares)
        # print("|\tCLOSE: ",priceOfShare)

    def sell(self, priceOfShare,DFindex):
        if self.shares <= 0.0: return
        elif self.lastTrans == False: return
        # Return to the bank, minus the brokerage
        self.bank = self.bank + self.shares*priceOfShare*(1-self.brokerage)
        self.transactions = self.transactions +1
        self.shares = 0.0
        self.lastTrans = False
        self.df.iloc[DFindex] = -1

    def setFitness(self, fitness):
        self.fitness = fitness

    def resetFitness(self):
        self.fitness = 0.0


    

    
    def __lt__(self, other):
         return self.fitness > other.fitness
    def __repr__(self):
        return str(self.fitness)
    def __str__(self):
        return str(self.fitness)


class evolver:
    def __init__(self,num_inputs = 1,num_hidden = 1, num_out = 1,pop_size = 50, template = None):
            self.executor = ThreadPoolExecutor(7)
            self.num_inputs =num_inputs
            self.num_hidden = num_hidden
            self.num_out = num_out
            self.pop_size = pop_size
            self.population = []
            for i in range(self.pop_size):
                self.population.append(node(self.num_inputs,self.num_hidden, template))
            self.setBanks(50.0)

    
    def setBanks(self, x):
        for i in range(len(self.population)):
            self.population[i].setBank(x)

    def propOne(self,i,inputLst, close, DFindex):
        
        signal = self.population[i].propogate(inputLst)
        if(signal > 0.5):  self.population[i].buy(close,DFindex,signal)
        elif(signal < -0.5): self.population[i].sell(close,DFindex)
        return signal
    
    def propogate(self, inputLst, close, DFindex):
        for i in range(self.pop_size):
           self.propOne(i,inputLst, close, DFindex)
            


    def calcFitness(self,close):
        # Use a softmax function to calculate fitness based on max bank
        fitnessFunc = lambda i: self.population[i].bank+(self.population[i].shares*close) #+self.population[i].transactions*2+ (-50 if self.population[i].transactions == 0 else 0)

        for i in range(self.pop_size):
            self.population[i].setFitness(fitnessFunc(i))
        self.population.sort()

    def repopulate(self):
        maxPops = math.ceil(len(self.population))
        # print("Max pops: ", maxPops)
        newPopulation = []
        while len(newPopulation) < len(self.population):
            for j in range(maxPops):
                if len(newPopulation) >= len(self.population): break
                newPopulation.append(self.population[0].mate(self.population[j]))
        self.population = newPopulation

