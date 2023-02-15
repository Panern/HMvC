#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: sometime in 2023
 |---| LastEditors: Wuji
 |---| LastEditTime: sometime in 2023
 |---| Description: graph filtering is implemented in this file
 |---| 
 |---| Copyright (c) 2023 by HMvC/AHMvC, All Rights Reserved. 
 
=========================================================================
'''

import numpy as np
import traceback
from high_order_graph import create_W
from scipy.sparse import csr_matrix

'''
=========================================================================
 
 |---| description: graph filtering for graph data is implemented in this function
 |---| param {*} X: the raw data
 |---| param {*} filter_order: the order of the graph filtering k
 |---| param {*} dtname: the name of the dataset
 |---| return {*}
    H: the filtered data
=========================================================================
'''
def graph_filtering_for_graph(X, filter_order=2, dtname='ACM') :
    try :
        A_list = []

        num_view = len(X) - 1
        N = X[0].shape[0]
        I = np.eye(N)
        if "Amazon" in dtname :
            H = X[:2].copy()
            A = X[2] + I
            D = np.sum(A, axis=1)

            D = np.power(D, -0.5)

            D[np.isinf(D)] = 0
            D = np.diagflat(D)
            A = D.dot(A).dot(D)
            L = I - A

            for i in range(num_view) :
                A_list.append(A)
                for k in range(filter_order) :
                    H[i] = (I - 0.5 * L).dot(H[i])
                print("filtering No. {}!!".format(i))
        else :
            print("Begin Filtering!")
            A_ = X[1 :].copy()
            H = []
            for i in range(num_view) :
                print("Begin Filtering {}!".format(i + 1))
                H.append(X[0])
                A = A_[i] + I
                D = np.sum(A, axis=1)

                D = np.power(D, -0.5)
                # D_[np.isinf(D_)] = 0
                # D_ = np.diagflat(D_)
                D[np.isinf(D)] = 0
                D = np.diagflat(D)
                A = D.dot(A).dot(D)
                L = I - A
                A_list.append(A)
                # _, Nbrs_list = find_subgraphs(A)
                # order_list = filter_node_order(Nbrs_list=Nbrs_list, L=L, dtname=dtname, num_view=i, X=X[0])

                for k in range(filter_order) :
                    H[i] = (I - 0.5 * L).dot(H[i])

                # for k in range(filter_order):
                #     H[i] = (I - 0.5 * L).dot(H[i])
                print("filtering No. {}!!".format(i))

        return H, A_list
    except Exception :
        traceback.print_exc()

Filtering = True
'''
=========================================================================
 
 |---| description: graph filtering for non-graph data is implemented in this function
 |---| param {*} X: the raw data
 |---| param {*} filter_order: the order of the graph filtering k
 |---| param {*} dtname: the name of the dataset
 |---| return {*}
    H: the filtered data
=========================================================================
'''
def graph_filtering_for_non_graph(X, filter_order=2, dtname='Cal101-7') :
    try :
        A_list = []
        if len(X) >= 10 :
            X = [X]
        num_view = len(X)

        W = create_W(X, order=1)
        N = X[0].shape[0]
        I = np.eye(N)

        print("Begin Filtering!")
        H = X.copy()

        for v in range(num_view) :
            A = W[v] + I
            D = np.sum(A, axis=1)

            D = np.power(D, -0.5)

            D[np.isinf(D)] = 0
            D = np.diagflat(D)
            A = D.dot(A).dot(D)
            # L = I - A

            A_list.append(A)
            if Filtering :
                for k in range(filter_order) :
                    H[v] = (0.5 * (I + W[v])).dot(H[v])

            print("filtering No. {}!!".format(v+1))


        return H, A_list
    except Exception :
        traceback.print_exc()




'''
=========================================================================
 
 |---| description: low-complexity low-pass filter for large attributed graphs with sparse adjacency matrix
 |---| param {*} X: raw feature matrix
 |---| param {*} A: the adjacency matrix
 |---| param {*} Dr: the degree vector
 |---| param {*} k1: the order of the filter
 |---| param {*} p: the parameter of the filter
 |---| return {*} H_low: the filtered feature matrix
 
=========================================================================
'''
def LowPassFilter_sparse(X, A, Dr, k1, p=0.5):
    N = X.shape[0]
    row_I = np.arange(N)
    col_I = np.arange(N)
    data_I = np.ones(N)
    
    # Sparse identity matrix
    I = csr_matrix((data_I, (row_I, col_I)), shape=(N, N))

    # Sparse degree matrix
    D = np.array(Dr)
    D = D + 1
    D = np.power(D, -0.5)
    D = csr_matrix((D, (row_I, col_I)), shape=(N, N))

    # self-loop and normalize
    S = A + I
    S = csr_matrix(S)
    # normalize
    S = D * S * D
    
    # filtered kernel
    # I - p * L_S = I - p * (I - S) = (1-p)I + pS
    F_M = (1-p) * I + p * S

    # filtered matrix with order k1
    H_low = F_M * X
    f_order = k1-1
    while f_order > 0:
        H_low = F_M * H_low
        f_order -= 1
    print("Filtering Done!")
    return H_low




Switcher_GF = {
        "graph" : graph_filtering_for_graph,
        "non-graph" : graph_filtering_for_non_graph
        }
