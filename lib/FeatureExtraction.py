#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:28:29 2018

@author: wentingyu
"""
import numpy as np
import cv2
from scipy.signal import convolve2d

def G_filter(x, y,sig_y_f, channel):
    x = np.int0(x)
    y = np.int0(y)
    if channel == 1:
        sig_x = 3
        sig_y = 1.5
    
    if channel == 2:
        sig_x = 4.5
        sig_y = 1.5
        
    if sig_y_f == True:
        f = 1/sig_y
    else:
        f = 1/sig_x
        
    M1 = np.cos(2*np.pi*f*(np.sqrt(x**2 + y**2)))
    
    G = (1/(2*np.pi*sig_x*sig_y))*(np.exp(-(x**2/sig_x**2 + y**2/sig_y**2)/2)) * M1
    return(G)


def getGKernel(channel,sig_y_f, size = 9):
    mask = np.zeros((size,size))
    
    for k_x in range(0,size):
        for k_y in range(0,size):
            mask[k_y, k_x] = G_filter(- size//2 + k_x,  - size//2 + k_y, sig_y_f, channel)
    return(mask)
  
    
def filterImage(eqImg, channel, size, sig_y_f):
    ROI = eqImg[0:48,]
    mask = getGKernel(channel,sig_y_f, size)
    mask = np.asanyarray(mask, np.float32)
    Fi = cv2.filter2D(ROI, -1, mask)         
    return(Fi)


def getFeatureVec(F1, F2):
    vec1 = []
    vec2 = []
    for y in range(0, 6):
        for x in range(0, 64):
            m1 = np.sum(F1[8*y:8*(y+1), 8*x:8*(x+1)])/64
            sigma1 = np.sum(np.abs(F1[8*y:8*(y+1), 8*x:8*(x+1)] - m1))/64
            
            m2 = np.sum(F2[8*y:8*(y+1), 8*x:8*(x+1)])/64
            sigma2 = np.sum(np.abs(F2[8*y:8*(y+1), 8*x:8*(x+1)] - m2))/64
            
            vec1.append(m1)
            vec1.append(sigma1)
            vec2.append(m2)
            vec2.append(sigma2)
    return(vec1 + vec2)


