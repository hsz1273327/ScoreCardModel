"""用于序列化Mixin,使用该Mixin的类可以用方法`obj.dumps()`来生成可以存储的一串字符串
也可以用类方法`clz.loads(dump_str)`来获取一个该类的实例对象.


>>> obj.dumps()
>>> clz.loads(dump_str)

"""
import dill
import base64


class SerializeMixin:
    """序列化模型Mixin
    """

    def dumps(self):
        """序列化模型

        Returns:

            str: - 序列化后的字符串,使用`dill`将对象序列化为bytes,再用`base64`将bytes转化为可以传输,可以存储的字符串.

        """
        return base64.b64encode(dill.dumps(self)).decode()

    @classmethod
    def loads(clz, dump_str: str):
        """反序列化模型

        Parameters:

            dump_str (str): - 序列化后的字符串

        Returns:

            clz: - 反序列化后返回该类型的实例对象

        """
        return dill.loads(base64.b64decode(dump_str))
