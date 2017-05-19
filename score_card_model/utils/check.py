__all__ = ["check_array_binary","check__array_continuous"]

import numpy as np
from sklearn.utils.multiclass import type_of_target
def check_array_binary(y:np.ndarray)->bool:
    '''
    检查一个array是否是二分的标签,可以是二值的字符串,二值的整数,或者布尔型数据
    '''
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        return False
    else:
        return True
        #raise ValueError('Label type must be binary')
def check__array_continuous(y:np.ndarray)->bool:
    """检查一个array不是连续型(浮点数)"""
    y_type = type_of_target(y)
    if y_type not in ['continuous']:
        return False
    else:
        return True
