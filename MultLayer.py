"""  This is a perceptron developed by Brandon TerLouw"""\

import NeuralNet as perc
import numpy as np
from math import exp
#import writeCsv as w

"""This method is what is called by another program to build and start
a neural network.  This is a feed forward neural netword with 1 hidden layer.
As parameters it takes the hidden number of nodes desired, the number of nodes
in the output, the data to be trained with and learning rate.  It returns two 
matrices, one between the input and hidden and one between the hidden and output.
Training will stop once the error per epoch is less than supplied value.
"""
def createAndTrain(hidNum,outNum,data,error,n):
    hidWeight = np.random.randint(0,10,(hidNum,len(data[1])))/1000.0
    outWeight = np.random.randint(0,10,((outNum,hidNum+1)))/1000.0
    (a,b,e) = epoch(hidWeight,outWeight,n,data)
    print(e)
    while(e > error):
        (a,b,e) = epoch(a,b,n,data)
        print("error for current epoch: " + str(e))
    return (a,b)  
    
"""
This method is where the majority of the neural network is built.
As parameters it takes a matrix of weights from input nodes to 
hidden nodes, a matrix of weights from hidden nodes to output nodes,
a learning rate and the data to be trained on.  It then updates and 
returns both matrices.  It also returns the error for this epoch.
"""    
def epoch(hidWeight,outWeight,n,data):
    learnRate = n
    e = 0
    for d in data:
        dCopy = d.copy()
        dCopy.append(1)
        dArray = np.array(dCopy[1:])
        #computes weights*inputs
        midValue = np.dot(hidWeight,dArray)
        #sigmoid function of the hidden layer value
        midSig = 1/(1+ np.exp(-midValue))
        midSig = np.append(midSig,[1])
        #computes weights*sigmoid of 
        finValue = np.dot(outWeight,midSig)
        finSig = 1/(1+ np.exp(-finValue))
        #computes error to return
        error = np.array(d[0]) - finSig
        error = error*error
        error = error/2
        e += np.sum(error)
        deltaArr = []
        #updates both matrices using back propagation
        for i in range(len(d[0])):
            delta = (finSig[i]-d[0][i])*finSig[i]*(1-finSig[i])
            deltaArr.append(delta)
            for j in range(len(outWeight[0])):
                outWeight[i][j] -=learnRate*delta*midSig[j]
        for i in range(len(hidWeight)):
            sum = 0
            for j in range(len(outWeight)):
                sum += deltaArr[j]*outWeight[j][i]
            derv = (1-midSig[i])*(1+midSig[i])
            dt = derv*sum
            for k in range(len(hidWeight[0])):
                hidWeight[i][k] -= dt*learnRate*dCopy[1+k]
            
    return (hidWeight,outWeight,e)