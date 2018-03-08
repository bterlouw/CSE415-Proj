import MultLayer as classifier
import csv
import random
from _sqlite3 import Row
import numpy as np
from pip._vendor.distlib.compat import raw_input

#imports data from iris data
def iris():
    data = []
    with open('irisData.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            curr = row.pop(4)
            temp = []
            if curr == 'setosa':
                temp.append([1, 0, 0])
                temp.extend([float(i) for i in row])
                data.append(temp)
            elif curr == 'versicolor':
                temp.append([0, 1, 0])
                temp.extend([float(i) for i in row])
                data.append(temp)
            else:
                temp.append([0, 0, 1])
                temp.extend([float(i) for i in row])
                data.append(temp)
    return data

#imports data from food data                
def food():
    data = []
    with open('FoodData.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            curr = row.pop(0)
            temp = []
            if curr == 'hotdog':
                temp.append([1, 0, 0])
                temp.extend([float(i) for i in row])
                data.append(temp)
            elif curr == 'bread':
                temp.append([0, 1, 0])
                temp.extend([float(i) for i in row])
                data.append(temp)
            else:
                temp.append([0, 0, 1])
                temp.extend([float(i) for i in row])
                data.append(temp)
    return data

#begins a session of using a neural net    
def run():
    a = str(raw_input("Iris data set?(i) or Food data set?(f)"))
    data = []
    if a=='i': data = iris()
    else: data = food()
    random.shuffle(data)
    g = int(raw_input("How many instances of test data?"))
    finalTest = []
    for i in range(g):
        finalTest.append(data.pop())
    nodes = int(raw_input("How many hidden nodes?"))
    error = float(raw_input("At what error per epoch to stop?"))  
    n = float(raw_input("What do you want training rate to be?")) 
    #using program to build and train neural net 
    (hidWeight,outWeight) = classifier.createAndTrain(nodes,3,data,error,n)
    #tests weights on all training data
    print("Testing training data:")
    testCor = 0
    testIncor = 0
    for d in data:
        d.append(1)
        midValueP = np.dot(hidWeight,np.array(d[1:]))
        midSigP = 1/(1+ np.exp(-midValueP))
        midSigP = np.append(midSigP,[1])
        finValueP = np.dot(outWeight,midSigP)
        finSigP = 1/(1+ np.exp(-finValueP))
        actual = []
        for i in range(len(finSigP)):
            if finSigP[i] > .5: actual.append(1)
            else: actual.append(0)
        if(actual==d[0]): testCor += 1
        else: testIncor +=1
    print(str(float(testCor/(testCor+testIncor)))+" fraction correct on test data")
    #tests weights on data never seen
    print("Testing never seen data:")
    newCor = 0
    newIncor = 0
    for d in finalTest:
        d.append(1)
        midValueP = np.dot(hidWeight,np.array(d[1:]))
        midSigP = 1/(1+ np.exp(-midValueP))
        midSigP = np.append(midSigP,[1])
        finValueP = np.dot(outWeight,midSigP)
        finSigP = 1/(1+ np.exp(-finValueP))
        m = max(finSigP)          
        actual = []
        for i in range(len(finSigP)):
            if finSigP[i] == m: actual.append(1)
            else: actual.append(0)
        if(actual==d[0]): newCor += 1
        else: newIncor +=1
    print(str(float(newCor/(newCor+newIncor)))+" fraction correct on new data")
    #allows user to enter own data(interesting to try with food)
    b = str(raw_input("Want to enter own data?(y/n)"))
    while (b=='y'):
        temp = []
        if a=='i':
            c = float(raw_input("Sepal length?"))
            d = float(raw_input("Sepal width?"))
            e = float(raw_input("Petal_length?"))
            f = float(raw_input("Petal_width?"))
            temp.append(c)
            temp.append(d)
            temp.append(e)
            temp.append(f)
            temp.append(1)
        else:
            c = float(raw_input("Calories?"))
            d = float(raw_input("Fat?"))
            e = float(raw_input("Protein?"))
            f = float(raw_input("Carbs?"))
            temp.append(c)
            temp.append(d)
            temp.append(e)
            temp.append(f)
            temp.append(1) 
        midValueP = np.dot(hidWeight,np.array(temp))
        midSigP = 1/(1+ np.exp(-midValueP))
        midSigP = np.append(midSigP,[1])
        finValueP = np.dot(outWeight,midSigP)
        finSigP = 1/(1+ np.exp(-finValueP))
        m = max(finSigP)          
        actual = []
        for i in range(len(finSigP)):
            if finSigP[i] == m: actual.append(1)
            else: actual.append(0)
        if a=='i':
            if actual == [1, 0, 0]: print('Setosa')
            elif actual == [0, 1, 0]: print('Versicolor')
            elif actual == [0, 0, 1]: print('Verginica')
            else: print('none')
        else:
            if actual == [1, 0, 0]: print('Hotdog')
            elif actual == [0, 1, 0]: print('Bread')
            elif actual == [0, 0, 1]: print('Peanut Butter')
            else: print('none')
        b = str(raw_input("Again?(y/n)"))
            



    
    
            