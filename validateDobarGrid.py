#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 20:52:06 2018

@author: fran
"""

import csv
from sklearn.metrics import accuracy_score, f1_score

'''============================================================
                        FRANOVO ZA VALIDACIJU
==============================================================='''

#na koliko PRVIH slika u test skupu radimo testiranje (--> na lastImg_index +1) ?
lastImg_index = 103


#stvaramo vektor predikcija
#predictions  = []
'''
with open('results.txt', 'rt') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        
        if(i>2*lastImg_index+1):
            break
        if (i%2==0):
            predictions.append(row)
        i = i+1
'''        

        
for k in range(76,98,2):   
    TreshMax=k    
    predictions  = []
    Name="results"+str(TreshMax)+"Average.txt"
    with open(Name, 'rt') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            if(i>2*lastImg_index+1):
                break
            if (i%2==0):
                predictions.append(row)
            i = i+1
    
    print(len(predictions))
    #stvaramo vektor ground truth labela
    groundTruth_labels = []
    with open('groundTruthInitialTestSet.csv', 'rt') as correct:
        spamreader = csv.reader(correct)
        i = 0
        for row in spamreader:
            if(i>lastImg_index):
                break
            groundTruth_labels.append(row)
            i = i+1
    
    acc = accuracy_score(groundTruth_labels, predictions, normalize=True, 
                                   sample_weight=None)
    f1macro = f1_score(groundTruth_labels, predictions, average='macro')
    f1micro = f1_score(groundTruth_labels, predictions, average='micro')
    f1weighted = f1_score(groundTruth_labels, predictions, average='weighted')
    #f1samples = f1_score(groundTruth_labels, predictions, average='samples')
    Name1="acc_f1"+str(TreshMax)+".txt"
    f =  open(Name1, 'w+') 
    f.write("Accuracy: %f\n" %acc)
    f.write("F1 score macro: %f\n" %f1macro)
    f.write("F1 score micro: %f\n" %f1micro)
    f.write("F1 score weighted: %f\n" %f1weighted)
    #f.write("F1 score: %f\n" %f1samples)
    f.close()
