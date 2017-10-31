"""
评分卡模型
------------

所谓评分卡其实只是一种用于表现预测概率的方式,书上说是用于给用户信用打分的.
通常它会结合logistic回归使用.当然了,个人认为用其他分类器也是一样的.

连续特征往往是非线性的,因此预测模型有两种思路

* 先做一些特征工程,通常工业上会将连续数据分段,把它转化为一组离散数据.
    如何分段也就是这种思路下预测模型准确度的根本了.

* 使用深度学习,深度学习自带特征抽象能力.

本模型就是第一种思路的实践.



模块设计
----------------

本模块有如下几个子模块:

* 连续数据离散化模块 discretization
* 计算离散数据各枚举值的证据权重 weight_of_evidence
* 分类器模型 models,其中现在只有logistic回归一个实现.
* 用于计算得分的评分卡模型 score_card


依赖
--------
本模块依赖numpy,sklearn,pandas


用法
--------
任何机器学习模型,使用流程都差不多

* 清洗特征,本模型不管
* 观察数据,使用`Discretization`来将连续数据离散化. 
* 将离散化后的数据每一列都使用`WeightOfEvidence`来计算离散数据的证据权重,每一项计算出来都可以计算出一个iv值(信息值),都算得了可以比较了挑大的选出来使用.
* 将离散数据都用`WeightOfEvidence`计算出每项的权重替换这些分类,然后拿这个放入分类器训练
* 重复上面的步骤找到效果最好的一个模型.
* 初始化一个`ScoreCardModel`用于计算最终得分


举例:

>>> from ScoreCardModel.discretization imnport Discretization
>>> from ScoreCardModel.weight_of_evidence imnport WeightOfEvidence
>>> from ScoreCardModel.models.logistic_regression import LogisticRegressionModel
>>> from ScoreCardModel.score_card import ScoreCardModel


"""
