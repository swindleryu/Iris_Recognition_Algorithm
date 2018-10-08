#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:02:44 2018

@author: wentingyu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IrisLocalization import *

def drawIris(filename):
    # draw Iris Localization on a single file 
    img = cv2.imread(filename,0) 
    cir_img = img.copy()
    [X_pupil,Y_pupil,r] = pupilBoundary(img)
    [X_iris, Y_iris, r_iris] = irisBoundary(img)
    cv2.circle(cir_img,(X_pupil,Y_pupil),r,(0,0,255),2)
    cv2.circle(cir_img,(X_iris,Y_iris),r_iris,(0,0,255),2)
    plt.imshow(cir_img, cmap = 'gray')
    plt.show()
    return


def drawAllIris(filenames, train = True):
    if train == True:
        for i in range(0, np.int0(np.size(filenames)/3)):
            plt.suptitle('{:01}'.format(np.int0(i + 1)) + 'th person')
            plt.subplot(1,3,1), drawIris(filenames[np.int0(3*i)])
            plt.subplot(1,3,2), drawIris(filenames[np.int0(3*i + 1)])
            plt.subplot(1,3,3), drawIris(filenames[np.int0(3*i + 2)])
       
    else:
        for i in range(0, np.int0(np.size(filenames)/4)):
            plt.suptitle('{:01}'.format(np.int0(i + 1)) + 'th person')
            plt.subplot(1,4,1), drawIris(filenames[np.int0(4*i)])
            plt.subplot(1,4,2), drawIris(filenames[np.int0(4*i + 1)])
            plt.subplot(1,4,3), drawIris(filenames[np.int0(4*i + 2)])
            plt.subplot(1,4,4), drawIris(filenames[np.int0(4*i + 3)])