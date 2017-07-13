#coding:utf-8

import numpy as np

def split(x, n):
    '''
    计算出每位数据所处的百分位数区间位置,默认分为5段,段数从1开始计数,数值越大越接近100%
    '''
    #nr = list(reversed(n))
    last = len(n)
    result = []
    for i in x:
        for k,j in enumerate(n):
            if i<j:
                result.append(k+1)
                break
        else:
            result.append(last+1)
    return np.array(result)
