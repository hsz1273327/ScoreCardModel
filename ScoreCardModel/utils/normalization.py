#coding:utf-8
def simple(x,n=1):
    """x取值范围[0,+inf)
        return: 取值为[0,1]
    """
    if x < 0:
        raise AttributeError("x must >0")
    return 1-(n/(n+x))
