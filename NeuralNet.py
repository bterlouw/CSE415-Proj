"""  This is a perceptron developed by Brandon TerLouw"""
from math import exp
global c
c = .1

def train(data):
    curr = 1
    weights = []
    theta = []
    theta.append(0)
    for w in range(len(data[0])):
        weights.append(0)
    while curr > .001:
        curr = epoch(weights,data,.5,theta)
    return [weights, theta[0], .5]
        

def epoch(weights, data, epsilon,theta):
    sum = 0
    for d in data:
        #0 is x-
        #1 is x+
        curr = BinaryResponse(weights, d,epsilon,theta)
        if curr != d[0]:
            if d[0] == 0: sum += downWeight(weights,d,theta)
            else: sum += upWeight(weights,d,theta)
    return sum

def BinaryResponse(weights, characteristics, epsilon,theta):
    sum = 0
    for n in range(1,len(weights)):
        sum += weights[n]*characteristics[n]
    if sum > (theta[0] - epsilon): return 1
    else: return 0
    
def SigResponse(Weights, characteristics, epsilon,theta):
    sum = 0
    for n in range(1,len(weights)):
        sum += weights[n]*characteristics[n]
    sig = 1/(1+math.exp(-sum))
    
        
    
def downWeight(weights,d,theta):
    global c  
    max = 0  
    for w in range(1,len(weights)):
        a = d[w]*c
        if a > max: max = a
        weights[w] -= d[w]*c
    theta[0] += c
    return max

def upWeight(weights,d,theta):
    global c
    max = 0
    for w in range(1,len(weights)):
        a = d[w]*c
        if a > max: max = a
        weights[w] += d[w]*c
    theta[0] -= c
    return max
        