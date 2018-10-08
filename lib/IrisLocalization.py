#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 23:48:16 2018

@author: wentingyu
"""
import numpy as np
import cv2
from scipy.spatial.distance import *

def roughPupilBound(img):
    # adding a threshold and small bounding box to exclude the eyelashes
    # so that we can get a more accurate estimates of Xp and Yp
    ret, thresh0 = cv2.threshold(img[70:, 62:-62],65,255,cv2.THRESH_BINARY)
    ver_pro = np.sum(thresh0, axis = 0)  # vertical projection profile, col sum, 320
    hor_pro = np.sum(thresh0, axis = 1)  # horizontal projection profile, row sum, 280
    Xp = np.argmin(ver_pro) # minima of vertical projection - vertical center of pupil
    Yp = np.argmin(hor_pro) # minima of vertical projection - horizontal center of pupil
    return(Xp + 62 ,Yp + 70)



def pupilBoundary(img):
    [Xp,Yp] = roughPupilBound(img)

    subimg = img[(Yp-62):(Yp+62), (Xp-64):(Xp+70)]
    ret, thresh = cv2.threshold(subimg,65,255,cv2.THRESH_BINARY)
    Xp_acc,Yp_acc = roughPupilBound(thresh)
    
    # calculating radius of the pupil
    dia1 = (np.max(np.where(thresh[Yp_acc, ] ==0)) - np.min(np.where(thresh[Yp_acc, ] ==0)))
    dia2 = (np.max(np.where(thresh[:,Xp_acc] ==0)) - np.min(np.where(thresh[:, Xp_acc] ==0)))
    radius = np.int0(np.sum([dia1,dia2])/4)
    
    imgm = cv2.GaussianBlur(img,(5,5),0)
    testimg = cv2.normalize(imgm, 242, 255)
    edge = cv2.Canny(testimg,0, 1,L2gradient = True)[72:,(Xp-75):(Xp+75)]


    circle = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT,1,250,
                           param1 =50,param2=10,minRadius= radius - 3,maxRadius =  radius + 4)

    circle = np.int0(np.array(circle))
    [xp,yp,r] = circle[0,0]

    return(xp + (Xp-75), yp + 72,r)
 

def irisBoundary(img):
    [Xp,Yp,r] = pupilBoundary(img)
    imgm = cv2.GaussianBlur(img,(5,5),0)
 
    circles = []
    rmID = []
    threshList = list(range(110,225,1))
    for i in range(0, len(threshList)):
        testimg = cv2.normalize(imgm, 110, threshList[i])
        edge = cv2.Canny(testimg,0, 1, L2gradient = True) 
        c = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT,1,250,
                           param1=30,param2=10,minRadius= 98,maxRadius= 118)  
        
        if c is None:
            rmID.append(i)
        else:
            circles.append(c[0,0])
 
    [X_iris, Y_iris, r_iris] = getBestCircle(circles, pupilBoundary(img))
    return(X_iris, Y_iris, r_iris)
    
    
def getBestCircle(CircleCenterList, pupilCenter):
    # choosing the circle that is closer to the center of pupil
    a = CircleCenterList
    b = pupilCenter
    dis = []
    for i in range(0, len(a)):
        dis.append(sqeuclidean(a[i][0:2], b[0:2]))
    return(a[np.argmin(dis)])
