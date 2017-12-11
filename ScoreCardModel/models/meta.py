"""定义分类器模型的抽象基类

"""
import abc
import numpy as np


class Model(abc.ABC):
    """模型的抽象类
    """
    _model = None
    feature_order = None

    def _predict(self, x):
        """二分类预测

        Parameters:

            x (Sequence): - 用于预测的特征向量

        Returns:

            bool: - 返回0,1也就是False/True,True表示预测值为True,否则说明预测值为False


        """
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        result = self._model.predict(x)
        return result

    def _predict_proba(self, x):
        """
        不同预测值的概率

        Parameters:

            x (Sequence): - 用于预测的特征向量

        Returns:

            float: - 预测值为False的概率
            float: - 预测值为True的概率


        """
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        result = self._model.predict_proba(x)
        return result

    @abc.abstractmethod
    def predict(self, x):
        """输入一个特征向量预测

        """
        pass

    @abc.abstractmethod
    def pre_trade(self, x):
        """向量预处理

        """
        pass

    @abc.abstractmethod
    def pre_trade_batch(self, x, y):
        """全部数据预处理

        """
        pass

    @abc.abstractmethod
    def _train(self, dataset, target, **kwargs):
        """训练一组训练数据

        """
        pass

    def train(self, dataset, target, *, test_size=0.3, random_state=0, **kwargs):
        """训练一组数据,输入必须是pandas的DataFrame

        Parameters:

            dataset (pandas.DataFrame): - 训练用的DataFrame
            target (Option[str,pandas.seri]): - 标签数据所在的列 
            test_size (float): - 测试集比例
            random_state: - 随机状态


        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, precision_score
        if isinstance(target, str):
            y = dataset[target].values
            columns = list(dataset.columns)
            columns.remove(target)
            self.feature_order = columns
            X_matrix = dataset[columns]
        else:
            y = target
            self.feature_order = list(dataset.columns)
            X_matrix = dataset
        X_matrix = self.pre_trade_batch(X_matrix, target)
        X_train, X_test, y_train, y_test = train_test_split(
            X_matrix, y, test_size=test_size, random_state=random_state)
        model = self._train(X_train, y_train, **kwargs)
        predictions = model.predict(X_test)
        print(model.score(X_test, y_test))
        print(precision_score(y_test, predictions, average='macro'))
        print(classification_report(y_test, predictions))
        self._model = model
