#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-04-26 17:58:27
 |---| LastEditors: panern
 |---| LastEditTime: 2023-02-15 10:12:01
 |---| Description: 
 |---| 
 |---| Copyright (c) 2023 by WuJi, All Rights Reserved. 
 
=========================================================================
'''
import numpy as np


'''
=========================================================================
 
 |---| description: this function is used to find the lower bound of the value in the array p
 |---| param {*} p: the probability array
 |---| param {*} rd: the random value
 |---| return {*}
    l: position of the lower bound
=========================================================================
'''
def lower_bound(p, rd):
    l = 0
    r = len(p) - 1
    while(l < r):

        mid = (l + r) // 2
        if(p[mid] > rd):
            r = mid
        else:
            l = mid + 1

    return l


'''
=========================================================================
 
 |---| description: this function is used to sample the nodes
 |---| param {*} A: the adjacency matrix
 |---| param {*} m: the number of nodes to be sampled
 |---| param {*} alpha: the parameter of the probability, fixed to 4
 |---| return {*}
    ind: the index of the sampled nodes
=========================================================================
'''
def node_sampling(A, m, alpha=4):

    if(len(A) > 10):
        D = np.sum(A + (A.dot(A)), axis=1).flatten()
    else:
        D = np.sum(A[0]+ (A[0].dot(A[0])), axis=1).flatten()
        for avv in A[1:]:
            D += np.sum(avv + (avv.dot(avv)), axis=1).flatten()
    print("D done!!")
    # if(len(np.shape(D)) > 1):
    #     D = D.A[0]
    #     print(1)

    D = D**alpha
    Max = np.max(D)
    Min = np.min(D)
    D = (D- Min) / (Max - Min)

    tot = np.sum(D)
    p = D / tot
    for i in range(len(p) - 1):
        p[i + 1] = p[i + 1] + p[i]
    ind = []
    vis = [0] * len(D)
    while(m):
        # print(m)
        while(1):
            # sd = 1
            # np.random.seed(m+sd)
            rd = np.random.rand()
            pos = lower_bound(p, rd)
            if(vis[pos] == 1):
                # sd += 1000
                continue
            else:
                vis[pos] = 1
                ind.append(pos)
                m = m - 1
                break

    print("Sampling done!")
    return ind


if __name__ == '__main__':
    from load_data import Amazon_photos
    
    X, gnd = Amazon_photos()
    node_sampling(X[2], 50)