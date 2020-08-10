from schemes.models import MLModel
from schemes.models import Serializable
from utils import is_array


class PublicKey(Serializable):
    ext_ = 'pk'

    def __init__(self, group=None, h1=None, h2=None, source=''):
        if source:
            self.fromFile(source)
            return
        assert group
        assert h1
        assert is_array(h1)
        assert is_array(h2)
        assert len(h1) == len(h2)
        self.group = group
        self.n = len(h1)
        self.h1 = h1
        self.h2 = h2


class MasterKey(PublicKey):
    ext_ = 'msk'

    def __init__(self, pk=None, s=None, t=None, source=''):
        if source:
            self.fromFile(source)
            return
        super(MasterKey, self).__init__(pk.group, pk.h1, pk.h2)
        assert s
        assert is_array(s)
        assert len(s) == len(pk.h1)
        assert t
        assert is_array(t)
        assert len(t) == len(pk.h1)
        self.s = s
        self.t = t


class DecryptionKey(Serializable):
    ext_ = 'dk'

    def __init__(self, model=None, skf=None, source=''):
        if source:
            self.fromFile(source)
            return
        assert model
        assert skf
        assert isinstance(model, MLModel)
        assert is_array(skf)
        classes = len(skf)
        assert classes
        self.classes = classes
        self.model = model
        self.skf = skf

    def serialize(self):
        model = self.model
        del self.model
        self.model_ = model.serialize()
        return super(DecryptionKey, self).serialize()

    def deserialize(self, o):
        super(DecryptionKey, self).deserialize(o)
        self.model = MLModel().deserialize(self.model_)
        del self.model_
        return self
