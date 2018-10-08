#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:31:14 2018

@author: wentingyu
"""

from IrisNormalization import *
from IrisLocalization import *
from ImageEnhancement import *
from FeatureExtraction import *
import cv2
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA


def getFilenames( train = True):
    names = []
    folderName = '../data/CASIA Iris Image Database (version 1.0)/'
     
    if train == True:
        numImg = 3
        index = 1
    else: 
        numImg = 4
        index = 2     
    for i in range(1,109):
        for j in range(1,np.int0(numImg + 1)):       
            fileName = folderName + '{:03}'.format(i) + '/'+ '{:01}'.format(index) +'/' + '{:03}'.format(i) +'_' +'{:01}'.format(index)+'_' + '{:01}'.format(j) +'.bmp'
            names.append(fileName)

    return(names)




def getLabel(rot, train = True):
    if train == True:
        k = 3*rot
    else: 
        k = 4
    return(list(np.repeat(np.arange(1,109),k)))



def getALLFeatures(filenames, Train = True, rot = True, size = 9):
    vec = []
    
    if Train == True and rot == True:
        # based on many trials, the following degress gives best accuracy
        # but in the final main function, we tried many other degrees 
        # and save the outfut for convinience
        rotDegrees = [-3,-2,-1,0,1,2,3] 
    else:
        rotDegrees = [0]
            
    for fname in filenames:
        img = cv2.imread(fname,0) 
        # Iris Localization
        [X_pupil,Y_pupil,r] = pupilBoundary(img)
        [X_iris, Y_iris, r_iris] = irisBoundary(img)
        
        for i in range(0,len(rotDegrees)):
            # Iris Nomalization & Enhancement
            normImg = Lima_Normalization(img, X_iris, Y_iris, r_iris, X_pupil,Y_pupil,r,rotDegrees[i])
            eqImage = hisEqulization(normImg, bg_Subtract = False)
            # Feature Extraction
            F1 = filterImage(eqImage, channel = 1, size = size, sig_y_f = True)
            F2 = filterImage(eqImage, channel = 2, size = size, sig_y_f = True)
            V = getFeatureVec(F1, F2)
            vec.append(V)
   
    return(vec)

def nearestCenterClassifier(trainFi, testF, Y_test, distance, rot = True):
    pred_testLabel = []
    rowsOfTest = np.shape(testF)[0]
    if rot == True:
        k = 7
    else:
        k = 1
    rowsOfTrain_reduced = np.int0(np.shape(trainFi)[0]/k)
    trainY_reduced = getLabel(1, train = True)
    
    for j in range(0,rowsOfTest):
        d = []  
        if distance == 1:
            di_min = list(map(lambda x,y: cityblock(x,y),trainFi, [testF[j]] * np.shape(trainFi)[0]))
            
        if distance == 2:
            di_min = list(map(lambda x,y: sqeuclidean(x,y),trainFi, [testF[j]] * np.shape(trainFi)[0]))
            
        if distance == 3:
            di_min = list(map(lambda x,y: cosine(x,y),trainFi, [testF[j]] * np.shape(trainFi)[0]))
        
        x = np.reshape(np.asarray(di_min), (rowsOfTrain_reduced, k))
        d = np.min(x, axis = 1)
        pred_testLabel.append(trainY_reduced[np.argmin(d)])
        
    
    accuracy = sum(pred_testLabel == np.array(Y_test))/len(Y_test)
        
    return(np.array(pred_testLabel),accuracy)



def combidePredbyMode(predLabel, k = 4):
    predMode =[]
    newLen = np.int0(len(predLabel)/k)  
    newtestLabel = np.arange(1,109)
    for i in range(0,newLen):
        predMode.append(mode(predLabel[i*k: k*(i + 1)])[0][0])
    accuracy = sum(predMode == newtestLabel)/len(newtestLabel)
    return(predMode, accuracy)

