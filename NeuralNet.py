"""  This is a 2 layer neural network developed by Brandon TerLouw"""
global c, theta
c = .1
theta = 0

def train(data):
    global theta
    curr = 1
    weights = []
    for w in range(len(data[0])):
        weights.append(0)
    while curr > .001:
        curr = epoch(weights,data,.01)
    return [weights, theta, .01]
        

def epoch(weights, data, epsilon):
    sum = 0
    for d in data:
        #0 is x-
        #1 is x+
        curr = BinaryResponse(weights, d,epsilon)
        if curr != d[0]:
            if d[0] == 0: sum += downWeight(weights,d)
            else: sum += upWeight(weights,d)
    return sum

def BinaryResponse(weights, characteristics, epsilon):
    global theta
    sum = 0
    for n in range(1,len(weights)):
        sum += weights[n]*characteristics[n]
    if sum > (theta - epsilon): return 1
    else: return 0
    
def downWeight(weights,d):
    global c, theta  
    max = 0  
    for w in range(1,len(weights)):
        a = d[w]*c
        if a > max: max = a
        weights[w] -= d[w]*c
        theta += c
    return max

def upWeight(weights,d):
    global c, theta
    max = 0
    for w in range(1,len(weights)):
        a = d[w]*c
        if a > max: max = a
        weights[w] += d[w]*c
        theta -= c
    return max
        