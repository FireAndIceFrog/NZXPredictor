import numpy as np
import pandas as  pd
import math
from numba import jit
from random import random

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
    mutation = 0.4
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
            
            if(random() > mutationChange ):
                mutate = child[layer][i] * (mutation*(random()*2-1))
                child[layer][i] = child[layer][i]+mutate
    return child







inputs = np.array([-0.1,-0.99])
# print(propogate(inputs,createLayers()))
model1 = createLayers(layers = 3,num_weights = 4)
model2 = createLayers(layers = 3,num_weights = 4)
print("-----Models--------")
print("--A--")
print(model1)
print("--B--")
print(model2)
print("-----Propogation--------")
print("A: ", propogate(inputs,model1,model1.shape[0],model1.shape[1],inputs.shape[0]))
print("B: ", propogate(inputs,model2,model2.shape[0],model2.shape[1],inputs.shape[0]))
print("-----------Mating---------")
print("Child: ",mate(model1, model2, model1.shape[0],model1.shape[1]))