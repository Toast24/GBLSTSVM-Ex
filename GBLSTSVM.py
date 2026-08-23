import csv
import time
import pandas as pd
import numpy as np
import json
import csv
# import sklearn.cluster.k_means_ as KMeans
from sklearn.cluster import KMeans
import warnings
from collections import Counter
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import collections
import copy
from sklearn.utils import shuffle

import pandas as pd

warnings.filterwarnings("ignore")  # ignore warning

def LSTWGBSVM(Data, TestX, c1, c2,):
    C1 = Data[Data[:, -1] == 1, :-1]
    C2 = Data[Data[:, -1] != 1, :-1]
    A = C1[:,:-1]
    B=C2[:,:-1]
    R1=C1[:,-1]
    R2=C2[:,-1]
    reg_term = 0.0001
    st = time.time()
    mat_A=np.array(A)
    mat_B=np.array(B)
    R_pos=np.array(R1)
    R_neg=np.array(R2)

    R_pos=np.asmatrix(R_pos)
    R_pos=np.transpose(R_pos)
    R_neg=np.asmatrix(R_neg)
    R_neg=np.transpose(R_neg)

    mat_e1 = np.ones((mat_A.shape[0], 1))
    mat_e2 = np.ones((mat_B.shape[0], 1))
    
    # define H=[A e1] , G=[B e2]
    mat_H = np.column_stack((mat_A, mat_e1))
    mat_G=np.column_stack((mat_B, mat_e2))
    mat_H_t = np.transpose(mat_H)
    mat_G_t = np.transpose(mat_G)
    mat_B_t=np.transpose(mat_B)
    mat_A_t=np.transpose(mat_A)
    mat_e1_t=np.transpose(mat_e1)
    mat_e2_t=np.transpose(mat_e2)
    m1=np.dot(mat_e1_t,mat_e1)
    m2=np.dot(mat_e2_t,mat_e2)
    
    # calculating the parameters for the first hyperplane
    
    
    inv_p_1 = np.linalg.inv((np.dot(mat_G_t, mat_G) + (1/c1) \
                             * np.dot(mat_H_t,mat_H)) + (reg_term * np.identity(mat_H.shape[1])))
    
    e2t_R_neg=np.dot(mat_e2_t,R_neg)   
    m2_e2t_R_neg= np.add(m2,e2t_R_neg)    
    e2_R_neg=np.add(mat_e2,R_neg)
    mat_B_t_e2_R_neg= np.dot(mat_B_t,e2_R_neg)   
    H=np.concatenate(( mat_B_t_e2_R_neg,m2_e2t_R_neg[0]),axis=0)
    hyper_p_1 = -1 * np.dot(inv_p_1,H)
    w1 = hyper_p_1[0:hyper_p_1.shape[0] - 1, :]
    b1 = hyper_p_1[-1, :]
    
    
    #calculating the parameter for the second hyperplane
    inv_p_2 = np.linalg.inv((np.dot(mat_H_t,mat_H) + (1 /c2) \
                      * np.dot(mat_G_t, mat_G)) + (reg_term * np.identity(mat_H.shape[1])))
    e1t_R_pos=np.dot(mat_e1_t,R_pos)   
    m1_e1t_R_pos= np.add(m1,e1t_R_pos)    
    e1_R_pos=np.add(mat_e1,R_pos)
    mat_A_t_e1_R_pos= np.dot(mat_A_t,e1_R_pos)   
    G=np.concatenate(( mat_A_t_e1_R_pos,m1_e1t_R_pos[0]),axis=0)
    hyper_p_2 =  np.dot(inv_p_2,G)
    w2 = hyper_p_2[0:hyper_p_2.shape[0] - 1, :]
    b2 = hyper_p_2[-1, :]
    
    
   
   

    m = TestX.shape[0]
    test_data = TestX[:, :-1]
    y1 = np.dot(test_data, w1) + b1 * np.ones(m)
    y2 = np.dot(test_data, w2) + b2 * np.ones(m)

    Predict_Y = np.sign(np.abs(y2) - np.abs(y1))

    no_test, no_col = TestX.shape
    err = 0.0
    Predict_Y = Predict_Y.T
    Predict_Y=Predict_Y.reshape(-1,1)
    obs1 = TestX[:, no_col-1]
    obs1=obs1.reshape(-1,1)
    for i in range(no_test):
        if Predict_Y[i] != obs1[i]:
            err += 1
    acc = ((TestX.shape[0] - err) / TestX.shape[0]) * 100
    et = time.time()
    elapsed_time = et - st

    return acc , elapsed_time

