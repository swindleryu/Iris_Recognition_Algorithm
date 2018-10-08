#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:20:07 2018

@author: wentingyu
"""
import numpy as np
import cv2

def bgIllumination(normImg):
    background = np.zeros((4,32))
    for y in range(0, 4):
        for x in range(0,32):
            background[y,x] = np.mean(normImg[16*y:16*(1+y),16*x:16*(x+1)])
    
    bg_resize = cv2.resize(background,(512,64), interpolation = cv2.INTER_CUBIC).astype(np.uint8)
    return(bg_resize)


def hisEqulization(normImg, bg_Subtract = True):
    equalImage = np.zeros((64,512))
    
    if bg_Subtract == True:
        bg_resize = bgIllumination(normImg)
        normImg_bg  = cv2.subtract(normImg,bg_resize)
    else:
        normImg_bg = normImg
        
    for y in range(0, 2):
        for x in range(0,16):
            img_slice = normImg_bg[y*32:(1+y)*32,x*32:(x+1)*32]
            equalImage[y*32:(1+y)*32,x*32:(x+1)*32] = cv2.equalizeHist(img_slice)
    return(equalImage)


