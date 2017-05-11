import numpy as np
import math
import copy

def calcShannonEnt(dataSet):
    labels = [i[-1] for i in dataSet]
    num = len(labels)
    uniqueSigs = set(labels)
    labelDict = {}
    for i in uniqueSigs:
        labelDict[i] = labels.count(i)
    shannonEnt = 0.0
    for i in uniqueSigs:
        prob = float( labelDict[i]) / num
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    values = [i[axis] for i in dataSet]
    nums = len(values)
    subDataSet = []
    for i in range(nums):
        if values[i] == value:
            temp = copy.deepcopy(dataSet[i])
            temp.pop(axis)
            subDataSet.append(temp)
    return subDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    nums = len(dataSet)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeatureIndex = -1
    for i in range(numFeatures):
        labels = [k[i] for k in dataSet]
        uniqueLabels = set(labels)
        newEntropy = 0.0
        for value in uniqueLabels:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet)) / nums
            newEntropy +=  prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatureIndex = i
    return bestFeatureIndex

def majorityCnt(classList):
    uniqueList = list(set(classList))
    maxCnt = 0
    maxClass = uniqueList[0]
    for i in uniqueList:
        tempCount = uniqueList.count(i)
        if tempCount > maxCnt:
            maxCnt = tempCount
            maxClass = i
    return maxClass

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeatureIndex]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeatureIndex])

    featureValues = [example[bestFeatureIndex] for example in dataSet]
    uniqueFeatureValues = set(featureValues)
    for feature in uniqueFeatureValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][feature] = createTree(splitDataSet(dataSet, bestFeatureIndex, feature), subLabels)
    return myTree

def classify(inputTree, featureLabels, testVec):
    firstKey = list(inputTree.keys())[0]
    secondDict = inputTree[firstKey]
    featureIndex = featureLabels.index(firstKey)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featureLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

