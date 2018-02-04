from .mixins.serialize_mixin import SerializeMixin
from .core import ScoreCardABC


class ScoreCardModel(ScoreCardABC, SerializeMixin):
    pass
