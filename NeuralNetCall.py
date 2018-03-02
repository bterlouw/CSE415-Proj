import NeuralNet as classifier
import csv
import random
from _sqlite3 import Row
data = []
with open('irisData.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        curr = row.pop(4)
        temp = []
        if curr == 'setosa':
            temp.append(0)
            temp.extend([float(i) for i in row])
            data.append(temp)
        elif curr == 'versicolor':
            temp.append(1)
            temp.extend([float(i) for i in row])
            data.append(temp)
random.shuffle(data)
finalTest = data.pop()
(weights,theta,epsilon) = classifier.train(data)
print("Testing training data:")
for i in range(len(data)):
    sum = 0
    for j in range(len(weights)):
        sum += weights[j]*data[i][j]
    if sum > (theta - epsilon):
        print("We compute 1, actually: "+str(data[i][0]))
    else:
        print("We compute 0, actually: "+str(data[i][0]))
print("Testing never seen data:")
sum = 0
for j in range(len(weights)):
    sum += weights[j]*data[i][j]
if sum > (theta - epsilon):
    print("We compute 1, actually: "+str(data[i][0]))
else:
    print("We compute 0, actually: "+str(data[i][0]))


        