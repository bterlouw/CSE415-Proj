"""  This is a perceptron developed by Brandon TerLouw"""
import NeuralNet as perc

def createAndTrain(hidNum,outNum,data):
    hidWeight = []
    for i in Range(hidNum):
        inputAr = []
        for j in Range(1,len(data[0])):
            inputAr.append(0)
        hidWeight.append(inputAr)
    outWeight = []    
    for i in Range(outNum):
        inputAr = []
        for j in Range(hidNum):
            inputAr.append(0)
        outWeight.append(inputAr)