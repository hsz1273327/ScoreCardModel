#coding:utf-8
from .core import Core
from .mixins import *

class ScoreCard(Core,ConstructorMixin,TrainMixin,SerializeMixin,
                ToDictMixin,CalculProbabilityMixin, CalculScoreMixin):
    pass
class ScoreCardWithKSDraw(ScoreCard,KSMixin,DrawMixin):
    pass
