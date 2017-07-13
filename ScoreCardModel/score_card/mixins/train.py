#coding:utf-8
from sklearn import cross_validation,metrics
import numpy as np
class TrainMixin(object):
    """需要有self._model_type"""
    def test_fit(self,times = 10,test_size=0.4, random_state=0,**kwargs):
        scores = []
        for i in range(times):
            X_train,X_test,y_train,y_test = cross_validation.train_test_split(self.woe_X,
                                                                              self.tag,
                                                                              test_size=test_size,
                                                                              random_state=random_state)
            model = self._model_type(**kwargs)
            model.fit(X_train,y_train)
            score = model.score(X_test,y_test)
            scores.append(score)

        X_train,X_test,y_train,y_test = cross_validation.train_test_split(self.woe_X,
                                                                          self.tag,
                                                                          test_size=test_size,
                                                                          random_state=random_state)
        model = self._model_type(**kwargs)
        model.fit (X_train,y_train)

        y_predict = model.predict(X_test)
        matrix = metrics.confusion_matrix(y_test, y_predict)
        return np.mean(scores),matrix

    def fit(self,name=None,**kwargs):
        model = self._model_type(**kwargs)
        model.fit(self.woe_X,self.tag)
        if name:
            self._model = model
        else:

            self._model = model
        return model

    def get_model_info(self):
        coef = self._model.coef_
        params = self._model.get_params()
        return {
            'coef':coef,
            'params': params
        }
