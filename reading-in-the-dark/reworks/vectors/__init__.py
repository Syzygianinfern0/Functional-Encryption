from exceptions import WrongInputError
from schemes.models import Serializable
from utils import is_array
from utils import is_scalar


class Vector(Serializable):
    ext_ = 'vec'

    def __init__(self, array=None, source=''):
        if source:
            self.fromFile(source)
            return
        if is_array(array):
            assert len(array) > 0, 'Trying to generate an image from an empty vector.'
            self.n = len(array)
            self.content = []
            for s in array:
                assert is_scalar(s), "Input doesn't contain valid scalars."
                self.content.append(int(s))
        else:
            WrongInputError(array)
