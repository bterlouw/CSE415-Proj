"""  This is a perceptron developed by Brandon TerLouw"""\

import NeuralNet as perc
import numpy as np
from math import exp
import writeCsv as w


def createAndTrain(hidNum,outNum,data,error,n):
    hidWeight = np.random.randint(0,10,(hidNum,len(data[1])))/1000.0
    outWeight = np.random.randint(0,10,((outNum,hidNum+1)))/1000.0
    (a,b,e) = epoch(hidWeight,outWeight,n,data)
    print(e)
    while(e > error):
        (a,b,e) = epoch(a,b,n,data)
        print("error for current epoch: " + str(e))
    return (a,b)
    
    
    
def epoch(hidWeight,outWeight,n,data):
    learnRate = n
    e = 0
    itr = 0
    for d in data:
        itr +=1
        dCopy = d.copy()
        dCopy.append(1)
        dArray = np.array(dCopy[1:])
        midValue = np.dot(hidWeight,dArray)
        midSig = 1/(1+ np.exp(-midValue))
        midSig = np.append(midSig,[1])
        finValue = np.dot(outWeight,midSig)
        finSig = 1/(1+ np.exp(-finValue))
        error = np.array(d[0]) - finSig
        error = error*error
        error = error/2
        e += np.sum(error)
        deltaArr = []
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