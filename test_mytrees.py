#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:30:02 2017

@author: jiangjinjin
"""

import numpy as np
import mytrees
import trees
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,0,'yes'],
               [0,1,'no'],
               [1,0,'no'],
               [0,0,'no']]
    labels = ['eyes', 'nose']
    return dataSet, labels

if __name__ == "__main__":
#    dataSet, labels = createDataSet()
#    print ("Shannon Entropy calculated by 'mytrees.py' is: {0:0.4f}".format(mytrees.calcShannonEnt(dataSet)))
#    print ("Shannon Entropy calculated by 'trees.py' is: {0:0.4f}".format(trees.calcShannonEnt(dataSet)))
#    print (mytrees.splitDataSet(dataSet, 0, 1))
#    print (mytrees.chooseBestFeatureToSplit(dataSet))
#    print (mytrees.createTree(dataSet, labels))
    
    fr = open("lenses.txt")
    lenses = [example.strip().split('\t') for example in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = mytrees.createTree(lenses, lensesLabels)
    print (lensesTree)