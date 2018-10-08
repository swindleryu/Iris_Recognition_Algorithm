#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:31:43 2018

@author: wentingyu
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

## Table output for LDA reduced performance and whole feature set performance
def getTable3(allPred,ldaPred):
    d = {'Original Feature Set': allPred, 'Reduced Feature Set': ldaPred}
    d = pd.DataFrame(d)
    d.index = ['L1 distance measure', 'L2 distance measure','Cosine similarity measure']
    return d


## Table output for LDA reduced performance and PCA performance
def getTable4(allPred,ldaPred):
    d = {'LDA Reduced Feature Set': allPred, 'PCA Reduced Feature Set': ldaPred}
    d = pd.DataFrame(d)
    d.index = ['L1 distance measure', 'L2 distance measure','Cosine similarity measure']
    return d


def drawCRRcurve(ncomp, accuracyList):
	plt.plot(ncomp, accuracyList)
	plt.title("CRR vs Dimensionality of the feature vector")# give plot a title
	plt.xlabel("Dimensionality of the feature vector")# make axis labels
	plt.ylabel("Correct Recognition Rate")
	plt.show()
