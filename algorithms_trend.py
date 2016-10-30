# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas
from scipy.stats import norm, mstats


def mk_test(ts, mode = 'simple', window = 50, alpha = 0.00001):
    """
    Mann-Kendall test for trend
    Detects linear trends (in pair with cox_stuart)
    Optimal window = 50

    Input:
        x:   a vector of data
        alpha: significance level (0.001 default)
        window: last n values, for finding trend
        mode: 
            - 'full' - if you need to know trend direction; 
            - 'simple' - if just trend existance.

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics 

    """
    x = [x[1] if x[1] else 0 for x in ts[-window:]]
    n = len(x)

    # calculate S 
    s = 0
    for k in xrange(n-1):
        for j in xrange(k+1,n):
            s += np.sign(x[j] - x[k])
    #s = [-1 if x[j] < x[k] else 1 for j in xrange(k+1,n) for k in xrange(n-1)]

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)
    # calculate the var(s)
    n = float(n)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
        print var_s
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in xrange(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test

    h = abs(z) > norm.ppf(1-alpha/2) 

    if (z<0) and h:
        trend = 'decreasing'
    elif (z>0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
    if mode == 'full':
        return trend, h, p, z
    else:
        return h, abs(z)


def cox_stuart(timeseries, window = 50, alpha = 0.0001, debug = False):
    """
    Cox-Stuart criterion
    H0: trend exists
    H1: otherwise

    Detects linear trends
    Optimal window = 50
    """
    n = window
    idx = np.arange(1,n+1)
    X = pandas.Series([x[1] if x[1] else 0 for x in timeseries[-n:]], index=idx)
    
    S1 = [(n-2*i) if X[i] <= X[n-i+1] else 0 for i in xrange(1,n//2)]
    n = float(n)
    S1_ = (sum(S1) - n**2 / 8) / math.sqrt(n*(n**2-1)/24)
    u = norm.ppf(1-alpha/2)
    if debug:
        print ('|S1*|:', abs(S1_))
        print ("u:",u)
    
    return abs(S1_) > u, abs(S1_) #H0 accept

def abbe_criterion(timeseries, window = 40, alpha = .00001, debug = False):

    """
    Abbe-Linnik criterion
    Detects exponential trend (in pair with autocorrelation)
    If window >50 => too much True
    """
    if len(timeseries) < window:
        return False
    series = pandas.Series([x[1] if x[1] else 0 for x in timeseries])
    a = series[-window:]
    mean = a.mean()
    

    X = zip(a[:-1],a[1:])
    s1 = sum([math.pow((x[0]-x[1]),2) for x in X])
    s2 = sum([pow((x-mean),2) for x in a])
    q = 0.5 * s1 / s2
    q_a = 0.6814
    n = len(a)
    Q = -(1-q)*math.sqrt((2*n+1)/(2-pow((1-q),2)))
    u = norm.ppf(alpha) #kobzar page 26
    if debug:
        print ('mean:', mean)
        print ('q:', q)
        print ('Q*:', Q)
        print ('U_alpha-1:', u)
    return Q < u, abs(Q)


def autocorrelation(timeseries, window = 50, alpha = .00001, debug = False):
    """
    Detects exponential trend

    """
    n = window
    idx = np.arange(1,n+1)
    series = pandas.Series([x[1] if x[1] else 0 for x in timeseries[-n:]], index=idx)

    X = zip(series[:-1],series[1:])
    sqr_sum = pow(sum(series),2)
    n_f = float(n)
    r = (n_f * sum([x[0]*x[1] for x in X]) - sqr_sum + n_f*series[1]*series[n]) / \
        (n_f * sum([x**2 for x in series]) - sqr_sum)
    
    r_ = abs(r + 1./(n_f-1.)) / math.sqrt(n_f*(n_f-3)/(n_f+1)/pow((n_f-1),2))
    u = norm.ppf(1-alpha/2)
    
    return abs(r_) > u, abs(r_)