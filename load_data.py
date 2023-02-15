#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: sometime in 2023
 |---| LastEditors: Wuji
 |---| LastEditTime: sometime in 2023
 |---| Description: this is for recording the running process
 |---| 
 |---| Copyright (c) 2023 by HMvC/AHMvC, All Rights Reserved. 
 
=========================================================================
'''


import os
import pickle as pkl
import sys
import warnings
from preprocess import load_dataset
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scanpy as sc
from collections import Counter
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")


def Amazon_photos() :
    X = []
    Amazon = load_dataset("Data/npz/amazon_electronics_photo.npz")
    Adj = sp.csr_matrix(Amazon.standardize().adj_matrix).A
    Attr = sp.csr_matrix(Amazon.standardize().attr_matrix).A
    Gnd = sp.csr_matrix(Amazon.standardize().labels).A
    Gnd = Gnd.T.squeeze()

    Attr = np.array(Attr)
    X.append(Attr)
    X.append(Attr.dot(Attr.T))
    X.append(np.array(Adj))
    return X, Gnd


def Amazon_photos_computers() :
    X = []
    Amazon = load_dataset("Data/npz/amazon_electronics_computers.npz")
    Adj = sp.csr_matrix(Amazon.standardize().adj_matrix).A
    Attr = sp.csr_matrix(Amazon.standardize().attr_matrix).A
    Gnd = sp.csr_matrix(Amazon.standardize().labels).A
    Gnd = Gnd.T.squeeze()
    Attr = np.array(Attr)
    X.append(Attr)
    X.append(Attr.dot(Attr.T))
    X.append(np.array(Adj))
    return X, Gnd


def Caltech101_7() :
    # data = Mp.loadmat('{}.mat'.format("./Data/mat/Caltech101-7"))
    mdata1 = sio.loadmat('./Data/mat/C_1_3.mat')
    mdata2 = sio.loadmat('./Data/mat/C_4_6.mat')
    mLabels = sio.loadmat('./Data/mat/C_label.mat')

    X = []
    # print(mdata1['data1'][0][0])
    X.append(np.array(mdata1['data1'][0][0]))
    X.append(np.array(mdata1['data2'][0][0]))
    X.append(np.array(mdata1['data3'][0][0]))

    X.append(np.array(mdata2['data4'][0][0]))
    X.append(np.array(mdata2['data5'][0][0]))
    X.append(np.array(mdata2['data6'][0][0]))

    gnd = np.squeeze(mLabels['labels'])


    return X, gnd


def Caltech101_20() :
    # data = Mp.loadmat('{}.mat'.format("./Data/mat/Caltech101-7"))
    mdata1 = sio.loadmat('./Data/mat/C_1_3_20.mat')
    mdata2 = sio.loadmat('./Data/mat/C_4_6_20.mat')
    mLabels = sio.loadmat('./Data/mat/C_label_20.mat')

    # print(mdata1.keys())
    # print(mdata2.keys())
    # print(mdata1['data1'].shape)
    X = []
    # print(mdata1['data1'][0][0])
    X.append(np.array(mdata1['data1'][0][0]))
    X.append(np.array(mdata1['data1'][1][0]))
    X.append(np.array(mdata1['data1'][2][0]))

    X.append(np.array(mdata2['data1'][0][0]))
    X.append(np.array(mdata2['data1'][1][0]))
    X.append(np.array(mdata2['data1'][2][0]))

    # for x in X:
    #     print(x.shape)
    gnd = np.squeeze(mLabels['labels'])

    # print(len(gnd))

    return X, gnd


def Citeseer():
    citation = sc.read("./Data/mtx/citeseer_cites.mtx")

    citation = sp.csr_matrix(citation.X).A

    citation[0][0] = 1
    # print(citation.shape)
    content = sc.read("./Data/mtx/citeseer_content.mtx")

    content = sp.csr_matrix(content.X).A
    # print(content.shape)

    labels = np.loadtxt("./Data/mtx/citeseer_act.txt", delimiter='\n')
    # print(len(labels))
    X = []

    content = np.array(content)
    X.append(content)
    X.append(citation)
    
    return X, labels




def Acm(dataname='ACM') :
    if dataname == "ACM" :
        # Load data
        dataset = "./Data/mat/" + 'ACM3025'
        data = sio.loadmat('{}.mat'.format(dataset))
        if (dataset == 'large_cora') :
            X = data['X']
            A = data['G']
            gnd = data['labels']
            gnd = gnd[0, :]
        else :
            X = data['feature']
            A = data['PAP']
            B = data['PLP']
            # C = data['PMP']
            # D = data['PTP']
    if sp.issparse(X) :
        X = X.todense()
    X_ = []
    A = np.array(A)
    B = np.array(B)
    X_.append(np.array(X))

    # for i in range(A.shape[0]):
    #     if A[i][i] == 1:
    #         A[i][i] = 0
    #     if B[i][i] == 1:
    #         B[i][i] = 0

    X_.append(A)
    X_.append(B)

    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X_, gnd


def Dblp() :
    ## Load data
    dataset = "./Data/mat/" + 'DBLP4057_GAT_with_idx'
    data = sio.loadmat('{}.mat'.format(dataset))
    if (dataset == 'large_cora') :
        X = data['X']
        A = data['G']
        gnd = data['labels']
        gnd = gnd[0, :]
    else :
        X = data['features']
        A = data['net_APTPA']
        B = data['net_APCPA']
        C = data['net_APA']
        # D = data['PTP']—

    if sp.issparse(X) :
        X = X.todense()
    X_ = []
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # for i in range(A.shape[0]):
    #     if A[i][i] == 1:
    #         A[i][i] = 0
    #     if B[i][i] == 1:
    #         B[i][i] = 0
    #     if C[i][i] == 1 :
    #         C[i][i] = 0
    X_.append(np.array(X))
    X_.append(A)
    X_.append(B)
    X_.append(C)
    # av.append(C)
    # av.append(D)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X_, gnd


def Imdb() :
    # Load data
    dataset = "./Data/mat/" + 'imdb5k'
    data = sio.loadmat('{}.mat'.format(dataset))
    if (dataset == 'large_cora') :
        X = data['X']
        A = data['G']
        gnd = data['labels']
        gnd = gnd[0, :]
    else :
        X = data['feature']
        A = data['MAM']
        B = data['MDM']
        # C = data['PMP']
        # D = data['PTP']
    if sp.issparse(X) :
        X = X.todense()
    X_ = []
    X_.append(np.array(X))
    A = np.array(A)
    B = np.array(B)

    X_.append(A)
    X_.append(B)
    # av.append(C)
    # av.append(D)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)
    gnd = np.squeeze(gnd)

    return X_, gnd


Switcher = {
        0 : Caltech101_7,
        1 : Caltech101_20,
        2 : Citeseer,
        3 : Acm,
        4 : Dblp,
        5 : Imdb,
        6 : Amazon_photos,
        7 : Amazon_photos_computers,
        }

if __name__ == "__main__" :
    _, gnd = Citeseer()
    print(len(np.unique(gnd)))
    pass
