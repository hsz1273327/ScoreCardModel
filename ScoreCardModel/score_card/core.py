
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin


class ScoreCardABC:
    """
    本模型需要使用一个已经训练好的分类器来初始化,预测,计算评分也都是依赖于它.

    Attributes:

        _model (ScoreCradModel.models.meta): - 训练好的预测模型
        b (int): - 偏置量的常数项,用于作为基数
        o (int): - 用于计算偏置量
        p (int): - 用于计算偏置量和因数项
        round_ (int): - 精度
        threshold (float): - 阈值,可选

    """

    def __init__(self, classifier, **kwargs):
        if isinstance(classifier, str):
            if classifier == "logistic":

                self._model = LogisticRegression(**kwargs)
            elif classifier == "bayes":
                pass
            else:
                raise RuntimeError("unknown classifier")

        elif isinstance(classifier, ClassifierMixin):
            self._model = classifier(**kwargs)

        else:
            raise RuntimeError(
                """classifier must in [logistic,bayes] 
                or a subclass of sklearn.base.ClassifierMixin"""
            )
        self.model_name = self.model.__name__

        

    def predict(self, x):
        """输入一个特征向量预测

        """
        pass

    def train(self, dataset, target, *, test_size=0.3, random_state=0, **kwargs):
        """训练一组数据,输入必须是numpy的数据集

        Parameters:

            dataset (numpy.dnarray): - 训练用的数据集
            target (numpy.dnarray): - 标签数据所在的列
            test_size (float): - 测试集比例
            random_state: - 随机状态


        """
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
        X_train, X_test, y_train, y_test = train_test_split(
            X_matrix, y, test_size=test_size, random_state=random_state)

        model.fit(X_matrix, y)
        predictions = model.predict(X_test)
        print(model.score(X_test, y_test))
        print(precision_score(y_test, predictions, average='macro'))
        print(classification_report(y_test, predictions))
        self._model = model
