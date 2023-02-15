#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: sometime in 2023
 |---| LastEditors: Wuji
 |---| LastEditTime: sometime in 2023
 |---| Description: run this file to reproduce the results in the paper
 |---| 
 |---| Copyright (c) 2023 by HMvC/AHMvC, All Rights Reserved. 
 
=========================================================================
'''

import numpy as np
from time import time
import math
from cvxopt import matrix, solvers
from sklearn.cluster import SpectralClustering, KMeans
from Metrics import metric_all
import yaml
import sys
import warnings
from logger_writting import Logger
import scipy.sparse as sp
from load_data import Switcher
from graph_filtering import Switcher_GF
from high_order_graph import create_W, create_F_W, create_W_anchor, Stable_graph
from anchor_selection import node_sampling

warnings.filterwarnings("ignore")


'''
=========================================================================
 
 |---| description: algorithm HMvC
 |---| param {*} H: filterd data
 |---| param {*} gnd: ground truth
 |---| param {*} dataname: the name of the dataset
 |---| param {*} order_W: the order of the graph
 |---| param {*} parameters: the parameters of the algorithm, \alpha, \beta, \mu
 |---| param {*} type: the type of the data, graph or non-graph
 |---| param {*} idd: the id of the parameters
 |---| return {*}
    acc, nmi, ari, f1, pur, time: accuracy, normalized mutual information, adjusted rand index, f1 score, purity, time
=========================================================================
'''
def HMvC(H, gnd, dataname, order_W=3, parameters=None, type=None, idd=0) :
    N = H[0].shape[0]
    I = np.eye(N)
    print("Data is {}. It contains {} nodes.".format(dataname, N))

    num_view = len(H)
    print("Datasets====>{} Views====>{}".format(dataname, num_view))

    if order_W <= 10 :
        Ls = create_F_W(H, order=order_W)
    else :
        Ls = Stable_graph(H)

    alpha_list = parameters["alpha"]
    beta_list = parameters['beta']
    mu_list = parameters['mu']
    alpha = float(alpha_list[idd])
    beta = float(beta_list[idd])
    mu = float(mu_list[idd])

    num_labels = len(np.unique(gnd))

    print("=============Initial!=============")
    # Set the range of filter order k
    S = []
    H_Ht = []
    for v in range(num_view) :
        H_Ht.append(H[v].dot(H[v].T))
    for v in range(num_view) :
        mid_va = np.linalg.inv(H_Ht[v] + I)
        S_v = mid_va.dot(H_Ht[v])
        S.append(S_v)

    print("=============Initial completed!=============")

    Lossss = []
    nada = [1 / num_view for vv in range(num_view)]
    S_ = S.copy()
    last_U = S.copy()
    nada_S = nada[0] * S_[0]
    for v in range(1, num_view) :
        nada_S += nada[v] * S_[v]

    U = nada_S.copy()

    loss_last = 0

    begin_time = time()
    for iter in range(20) :

        last_U = U.copy()
        Loss = 0

        for v in range(num_view) :
            re_loss = nada[v] * (np.linalg.norm(
                    H[v].T - H[v].T.dot(S_[v], ), ord='fro'
                    ) ** 2 + alpha * np.linalg.norm(
                    S_[v] - Ls[v], ord='fro'
                    ) ** 2)

            fusion_loss = beta * (np.linalg.norm(U - nada_S) ** 2) + mu * np.linalg.norm(U) ** 2
            Loss += re_loss + fusion_loss
        Lossss.append(Loss)
        oder = math.log10(Loss)
        oder = int(oder)
        oder = min(5, oder)
        Tol = 1 * math.pow(10, -oder)

        if math.fabs(Loss - loss_last) <= math.fabs(Tol * Loss) :
            break
        else :
            loss_last = Loss

        for v in range(num_view) :
            tmp1 = nada[v] * (H_Ht[v] + (alpha + beta) * I)
            tmp2 = nada[v] * (H_Ht[v] + alpha * Ls[v])
            tmp3 = beta * (U - nada_S + nada[v] * S_[v])
            S_[v] = np.linalg.inv(tmp1).dot(tmp2 + tmp3)


        nada_S = nada[0] * S_[0]
        for v in range(1, num_view) :
            nada_S += nada[v] * S_[v]


        U = (beta * nada_S) / (beta + mu)
        q = []
        P = np.zeros((num_view, num_view))
        for i in range(num_view) :
            for j in range(num_view) :
                P[i][j] = np.trace(np.multiply(S_[i], S_[j]))
            g_v = np.linalg.norm(H[i].T - H[i].T.dot(S_[i])) ** 2 + alpha * np.linalg.norm(
                    S_[i] - Ls[i]
                    ) ** 2
            q.append(g_v - 2 * beta * np.trace(U.dot(S_[i])))
        P = beta * P

        P = matrix(P)
        q = matrix(q)
        G = matrix(-1.0 * np.eye(num_view))
        h = matrix([0.0 for i in range(num_view)])
        A_eq = matrix([[1.0] for i in range(num_view)])
        B_eq = matrix([1.0])
        try :
            solvers.options['show_progress'] = False
            result_QP = solvers.qp(
                    P=P, q=q, G=G, h=h, A=A_eq, b=B_eq
                    )
            if result_QP['status'] != "optimal" :
                U = last_U.copy()
                break
            nada = np.array(result_QP['x']).T.copy()
            nada = np.squeeze(nada)
            nada = np.fabs(nada)
        except Exception :
            U = last_U.copy()
            break

    C = 0.5 * (np.fabs(U) + np.fabs(U.T))

    if type == 'graph' :
        SpC = SpectralClustering(n_clusters=num_labels, affinity='precomputed', random_state=21)
        predict_labels = SpC.fit_predict(C)
    else :
        u, s, v = sp.linalg.svds(C, k=num_labels, which='LM')
        kmeans = KMeans(n_clusters=num_labels, random_state=23).fit(u)
        predict_labels = kmeans.predict(u)

    end_time = time()

    Time = math.fabs(end_time - begin_time)

    re = metric_all.clustering_metrics(predict_labels, gnd)
    ac, nm, ari, f1, pur = re.evaluationClusterModelFromLabel()


    return ac, nm, ari, f1, pur, Time

'''
=========================================================================
 
 |---| description: algorithm AHMvC
 |---| param {*} H: filter data
 |---| param {*} gnd: ground truth
 |---| param {*} inds: index of data
 |---| param {*} dataname: name of dataset
 |---| param {*} order_W: order of W
 |---| param {*} num_anchor: number of anchor
 |---| param {*} parameters: parameters, \alpha, \beta, \mu
 |---| return {*}
    acc, nmi, ari, f1, pur, Time: accuracy, normalized mutual information, adjusted rand index, f1 score, purity, time
=========================================================================
'''
def HMvC_anchor(H, gnd, inds, dataname, order_W=3, num_anchor=100, parameters=None):
    alpha_list = parameters["alpha"]
    beta_list = parameters['beta']
    mu_list = parameters['mu']
    alpha_list = [float(alpha) for alpha in alpha_list]
    beta_list = [float(beta) for beta in beta_list]
    mu_list = [float(mu) for mu in mu_list]

    num_labels = len(np.unique(gnd))
    N = H[0].shape[0]
    I = np.eye(N)

    num_view = len(H)
    print("Datasets====>{} Views====>{}".format(dataname, num_view))

    B = []
    for h in H:
        B.append(h[inds])

    if order_W <= 1000:
        Ls = create_W_anchor(H, order=order_W, inds=inds)
    else:
        Ls = Stable_graph(H)
        Ls_new = []
        for Ls_v in Ls:
            Ls_new.append(Ls_v[inds])
        Ls = Ls_new.copy()



    print("=============Initial!=============")
    # Set the range of filter order k
    S = []
    B_Ht = []
    B_Bt = []
    Im = np.eye(num_anchor)

    for v in range(num_view):
        B_Ht.append(B[v].dot(H[v].T))
        B_Bt.append(B[v].dot(B[v].T))

    for v in range(num_view):
        mid_va = np.linalg.inv(B_Bt[v] + Im)
        S_v = mid_va.dot(B_Ht[v])
        S.append(S_v)

    print("=============Initial completed!=============")

    Best_alpha = 0
    Best_beta = 0
    Best_mu = 0
    Best_re = [0, 0, 0, 0, 0]
    Best_time = 0
    for alpha in alpha_list:
        for beta in beta_list:
            for mu in mu_list:

                nada = [1 / num_view for i in range(num_view)]
                S_ = S.copy()
                nada_S = nada[0] * S_[0]
                for v in range(1, num_view):
                    nada_S += nada[v] * S_[v]

                U = nada_S

                loss_last = 0
                losss = []

                begin_time = time()
                for iter in range(20):
                    last_S = S_.copy()
                    last_U = U.copy()
                    Loss = 0

                    for v in range(num_view):
                        re_loss = nada[v] * (np.linalg.norm(H[v].T - B[v].T.dot(S_[v]))
                                             ** 2 + alpha * np.linalg.norm(S_[v] - Ls[v]) ** 2)
                        fusion_loss = beta * \
                                      (np.linalg.norm(U - nada_S) ** 2) + mu * np.linalg.norm(U) ** 2
                        Loss += re_loss + fusion_loss

                    losss.append(Loss)


                    oder = math.log10(Loss)
                    oder = int(oder)
                    oder = min(5, oder)
                    Tol = 1 * math.pow(10, -oder + 1)
                    print(
                            "Iter===========>{}    Loss============>{}".format(
                                    iter, Loss))

                    if math.fabs(Loss - loss_last) <= math.fabs(Tol * Loss):
                        print("The convergence condition meet !!")
                        break
                    else:
                        loss_last = Loss
                    for v in range(num_view):
                        # tmp1 = nada[v] * (B_Bt[v] + (alpha + beta) * Im)
                        # tmp2 = nada[v] * (B_Ht[v] + alpha * Ls[v])
                        # tmp3 = beta * (U - nada_S + nada[v] * S_[v])
                        # S_[v] = np.linalg.inv(tmp1).dot(tmp2 + tmp3)

                        tmp1 = (B_Bt[v] + (alpha + nada[v] * beta) * Im)
                        tmp2 = B_Ht[v] + alpha * Ls[v]
                        tmp3 = beta * (U - nada_S + nada[v] * S_[v])
                        try:
                            S_[v] = np.linalg.inv(tmp1).dot(tmp2 + tmp3)
                        except Exception:
                            S_v = np.linalg.pinv(tmp1).dot(tmp2 + tmp3)

                    nada_S = nada[0] * S_[0]
                    for v in range(1, num_view):
                        nada_S += nada[v] * S_[v]

                    U = (beta * nada_S) / (beta + mu)
                    q = []
                    P = np.zeros((num_view, num_view))
                    for i in range(num_view):
                        for j in range(num_view):
                            P[i][j] = np.trace(np.multiply(S_[i], S_[j]))
                        g_v = np.linalg.norm(H[i].T - B[i].T.dot(S_[i])) ** 2 + alpha * np.linalg.norm(
                                S_[i] - Ls[i]) ** 2
                        q.append(g_v - 2 * beta * np.trace(U.dot(S_[i].T)))
                    P = beta * P

                    P = matrix(P)
                    q = matrix(q)
                    G = matrix(-1.0 * np.eye(num_view))
                    h = matrix([0.0 for i in range(num_view)])
                    A_eq = matrix([[1.0] for i in range(num_view)])
                    B_eq = matrix([1.0])
                    try:
                        solvers.options['show_progress'] = False
                        result_QP = solvers.qp(
                                P=P, q=q, G=G, h=h, A=A_eq, b=B_eq)

                        if result_QP['status'] != "optimal":
                            U = last_U.copy()
                            break
                        nada = np.array(result_QP['x']).T.copy()
                        nada = np.squeeze(nada)
                        nada = np.fabs(nada)
                    except Exception:
                        U = last_U.copy()
                        break



                D = np.sum(U, axis=1)
                D = np.power(D, -0.5)
                D[np.isinf(D)] = 0
                D[np.isnan(D)] = 0
                D = np.diagflat(D)  # (m,m)

                S_hat = D.dot(U)  # (m,n)

                S_hat_tmp = S_hat.dot(S_hat.T)  # (m,m)
                S_hat_tmp[np.isinf(S_hat_tmp)] = 0
                S_hat_tmp[np.isnan(S_hat_tmp)] = 0
                # sigma, E = scipy.linalg.eig(S_hat_tmp)
                E, sigma, v = sp.linalg.svds(
                        S_hat_tmp, k=num_labels, which='LM')
                sigma = sigma.T
                sigma = np.power(sigma, -0.5)
                sigma[np.isinf(sigma)] = 0
                sigma[np.isnan(sigma)] = 0
                sigma = np.diagflat(sigma)
                C_hat = (sigma.dot(E.T)).dot(S_hat)
                C_hat[np.isinf(C_hat)] = 0
                C_hat[np.isnan(C_hat)] = 0
                C_hat = C_hat.astype(float)
                
                
                kmeans = KMeans(n_clusters=num_labels, random_state=20).fit(C_hat.T)
                predict_labels = kmeans.predict(C_hat.T)


                end_time = time()

                Time = math.fabs(end_time - begin_time)
                re = metric_all.clustering_metrics(predict_labels, gnd)

                ac, nm, ari, f1, pur = re.evaluationClusterModelFromLabel()

                print(
                        "Alpha===>{}  Beta==>{}  Mu===>{}  ACC:===>{}\n".format(
                                alpha, beta, mu, ac))
                fh = open(
                        './re/Main_run/Anchor/HMvC_{}.txt'.format(
                                dataname, order_W), 'a')
                if ac > Best_re[0]:
                    Best_re[0] = ac
                    Best_re[1] = nm
                    Best_re[2] = ari
                    Best_re[3] = f1
                    Best_re[4] = pur
                    Best_alpha = alpha
                    Best_beta = beta
                    Best_mu = mu
                    Best_time =  Time
                fh.write(
                        "Time={}  alpha={} beta={} mu={} acc={} nmi={} ari={} f1={} pur={}".format(
                                Time,  alpha, beta, mu, ac, nm, ari, f1, pur))
                fh.write('\r\n')
                fh.flush()
                fh.close()

    print(
            "\nThe best result:ACC===>{}NMI===>{}ARI===>{}F1===>{}PUR===>{}\nIt's obtained at Alpha={} Beta={} Mu={} Time={}\n".format(
                    Best_re[0],
                    Best_re[1],
                    Best_re[2],
                    Best_re[3],
                    Best_re[4],
                    Best_alpha,
                    Best_beta,
                    Best_mu, Best_time))

    return Best_re, Best_alpha, Best_beta, Best_mu, Best_time


if __name__ == '__main__' :

    sys.stdout = Logger()
    sys.stdout.show_version()
    f = open('reproduce.yaml')
    cifg = yaml.load(f, Loader=yaml.FullLoader)
    parameters = cifg["parameters"]
    datasets = cifg["datasets"]

    for idx, data in enumerate(datasets) :
        typeD = "non-graph"
        order = 2
        X, gnd = Switcher[idx]()

        H, A_list = Switcher_GF[typeD](X)

        ac, nm, ari, f1, pur, Time = HMvC(
                H=H, gnd=gnd, dataname=data, order_W=order, parameters=parameters, type=typeD, idd=idx
                )
        print("Datasets======>{} ACC=======>{} NMI=======>{} Time=======>{}".format(data, ac, nm, Time))
