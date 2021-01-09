#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# __init__.py
# @Author : wanhanwan (wanshuai_shufe@163.com)
# @Link   : ~
# @Date   : 2019/11/24 上午9:53:13

import numpy as np
from pykalman import KalmanFilter as KF


def KFD(YY, n_iter=100):
# use kalman filter to denoise univariate time series
# YY is a T*1 array
# n_iter is the maximum iteration number in EM algorithm
    tL = len(YY)
    yy = np.reshape(YY, (tL, 1))
    obs_mat = np.ones((tL, 1, 1))
    initial_state_mean = np.array([YY.mean()])
    initial_state_covariance =np.array([np.abs(initial_state_mean)*1e4])

    kf = KF(em_vars=['transition_covariance', 'observation_covariance', 'transition_matrices'], \
            observation_matrices=obs_mat, \
            transition_offsets=np.array([0]), observation_offsets=np.array([0]), \
            initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance)

    kf = kf.em(yy, n_iter=n_iter)

    KF_res = {'Persistence': kf.transition_matrices, \
              'FilteredValues': kf.filter(yy)[0], \
              'SmoothedValues': kf.smooth(yy)[0]}

    return KF_res


def DLM(Y, X, intercept=0, n_iter=100):

# Estimate Dynamic Linear Model by KF
# n_iter is the maximum iteration number in EM algorithm
# Y is T-length array, target variable
# X is T*N array,  predictors
# Set intercept = 1 to include an intercept in regression

    if len(X.shape) == 1:
        X = np.reshape(X, (len(X), 1))

    tL, N = X.shape

    if intercept == 1:
        X = np.hstack((np.ones((tL, 1)), X))
        tL, N = X.shape


    yy = np.reshape(Y, (tL, 1))

    obs_mat = np.reshape(X, (tL, 1, N))
    transition_matrices = np.diag(np.ones(N))
    transition_offsets = np.zeros(N)
    observation_offsets = np.array([0])
    initial_state_mean = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)  # 用最小二乘作为初始值
    initial_state_covariance = np.diag(np.abs(initial_state_mean)*1e4)


    kf = KF(em_vars=['transition_covariance', 'observation_covariance', 'initial_state_covariance'], \
         transition_matrices=transition_matrices, observation_matrices=obs_mat, \
         transition_offsets=transition_offsets, observation_offsets=observation_offsets, \
         initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance)

    kf = kf.em(yy, n_iter=n_iter)

    KF_res = {'FilteredValues': kf.filter(yy)[0], 'SmoothedValues': kf.smooth(yy)[0]}

    return KF_res