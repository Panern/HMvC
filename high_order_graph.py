#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: sometime in 2023
 |---| LastEditors: Wuji
 |---| LastEditTime: sometime in 2023
 |---| Description: High-order graph construction is implemented in this file
 |---| 
 |---| Copyright (c) 2023 by HMvC/AHMvC, All Rights Reserved. 
 
=========================================================================
'''


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

'''
=========================================================================
 
 |---| description: Stable graph construction (the infinity-order similarity graph)
 |---| param {*} X: the raw data
 |---| return {*}
    W: the stable graph
=========================================================================
'''
def Stable_graph(X) :
    from sklearn.metrics.pairwise import cosine_similarity
    W = []

    for x in X :
        # print(type(x))
        # print(x.shape)

        S = cosine_similarity(x, x)
        S = (S + 1.) / 2

        S = S - np.eye(N=x.shape[0])
        D = np.sum(S, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        S = D.dot(S).dot(D)

        eig_vls, eig_vcs = np.linalg.eigh(S)

        R_idx = []
        r = 0

        for idx, eig_vl in enumerate(eig_vls) :
            # 1.0 is not equal to 1 with float datatype
            if 1 - eig_vl <= 1e-6 :
                R_idx.append(idx)
                r += 1
        if r == 0 :
            R_idx.append(np.argmax(eig_vls))
            r = 1
        # print("There are {} lambdas==1.\n".format(r))

        v_0 = eig_vcs[:, R_idx[0]].reshape(-1, 1)
        stable_S = v_0.dot(v_0.T)
        for r_idx in R_idx[1 :] :
            v_i = eig_vcs[:, r_idx].reshape(-1, 1)
            stable_S += v_i.dot(v_i.T)

        W.append(stable_S)

    return W


'''
=========================================================================
 
 |---| description: the mixed high-order graph construction
 |---| param {*} X: the raw data
 |---| param {*} order: the order of the graph
 |---| return {*}
    W: the mixed high-order graph
=========================================================================
'''
def create_F_W(X, order=2) :

    W = []

    for x in X :

        S = cosine_similarity(x, x)

        S = (S + 1.) / 2

        S = S - np.eye(N=x.shape[0])

        D = np.sum(S, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        S = D.dot(S).dot(D)
        S_tmp = S.copy()

        S_ = S.copy()

        for i in range(order - 1) :
            S_tmp = S_tmp.dot(S_)
            S += S_tmp

        W.append(S)

    return W


'''
=========================================================================
 
 |---| description: the high-order graph construction
 |---| param {*} X: the raw data
 |---| param {*} order: the order of the graph
 |---| return {*}
    W: the high-order graph
=========================================================================
'''
def create_W(X, order=1) :

    W = []

    for x in X :
        # print(type(x))
        # print(x.shape)

        S = cosine_similarity(x, x)
        S = (S + 1.) / 2

        S = S - np.eye(N=x.shape[0])
        D = np.sum(S, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        S = D.dot(S).dot(D)

        S_ = S.copy()

        for i in range(order - 1) :

            S = np.dot(S, S_)

        W.append(S)
    return W
    pass


'''
=========================================================================
 
 |---| description: this is for constructing the high-order anchor graph 
 |---| param {*} H: the raw data
 |---| param {*} order: the order of the graph
 |---| param {*} inds: the anchor nodes' indeces
 |---| return {*}
    W: the high-order anchor graph 
=========================================================================
'''
def create_W_anchor(H, order=2, inds=None) :

    W = []
    # I = np.eye(X[0].shape[0])
    N = H[0].shape[0]
    all = [i for i in range(N)]
    for h in H :

        S = cosine_similarity(h, h)
        S = (S + 1.) / 2
        S = S - np.eye(N)
        D = np.sum(S, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        S = D.dot(S).dot(D)

        A1_ = S[inds]
        A1 = S[inds]
        A2 = S[list(set(all) - set(inds))]

        for od in range(order - 1) :
            A1 = np.concatenate((A1.dot(A1_.T), A1.dot(A2.T)), axis=1)

        W.append(A1)

    return W
