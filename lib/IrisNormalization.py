#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 23:53:53 2018

@author: wentingyu
"""
import numpy as np

def Lima_Normalization(oriImg, Xi, Yi, ri, Xp, Yp, rp, rot = 0, M = 64, N = 512):
    In = np.zeros((M,N))
    
    #l is the distance of two centers
    l = np.sqrt((Xi - Xp)**2 + (Yi - Yp)**2)
    
    if l == 0: 
        alpha = 0 # alpha is the angle bw two centers 
    else:
        alpha = np.arcsin(np.abs(Yi - Yp)/l)
    
    for X in range(0, N):
        theta = 2*np.pi*X/(N-1) 
        beta = np.pi - theta - alpha
        
        # r3: is the changing distance 
        r3 = l*np.cos(beta) + np.sqrt((np.cos(beta)**2 - 1)*l**2 + ri**2)
        
        # Normalized Pupil boundary Location
        Xp_theta = Xp + np.cos(theta)*rp
        Yp_theta = Yp - np.sin(theta)*rp
    
        # Normalized Iris boundary location
        Xi_theta = Xp + np.cos(theta)*r3
        Yi_theta = Yp - np.sin(theta)*r3

        for Y in range(0,M):
            x = np.int0(Xp_theta + (Xi_theta - Xp_theta)*Y/(M-1))
            y = np.int0(Yp_theta + (Yi_theta - Yp_theta)*Y/(M-1))
            
            # Additional Conditions if iris out of boundary
            x = min(x, 319) or max(0,x)
            y = min(y, 279) or max(0,y)
            
            In[Y, X] = oriImg[y,x]
    
    # In1 is the new normalized image with rotation
    In1 = np.zeros((M,N))
    # rot imamges
    X0 = np.int0((N-1)*rot/(360)) 
    
    if X0 > 0 :
        In1[:,0:(N-X0)] = In[:,X0:N]
        In1[:,(N-X0):N] = In[:,0:X0]
    else:
        In1[:, 0:-X0] = In[:,(N+X0):N]
        In1[:, -X0:N] = In[:,0:(N+X0)]

    return(In1.astype(np.uint8))
        
