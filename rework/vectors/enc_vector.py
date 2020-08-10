from objects.models import Serializable
from utils import batch_exp
from utils import is_array


class EncryptedVector(Serializable):
    ext_ = 'evec'

    def __init__(self, group=None, simplifier=None, left=None, right=None, source=''):
        if source:
            self.fromFile(source)
            return
        assert left
        assert right
        assert group
        assert simplifier
        assert is_array(left)
        assert is_array(right)
        assert len(left) == len(right), 'Ciphertext was not properly generated.'
        assert len(left) > 1, 'Ciphertext is empty.'
        assert (is_array(x) for x in left)
        assert (is_array(x) for x in right)
        assert (len(x) == 2 for x in left)
        assert (len(x) == 2 for x in right)
        self.n = len(left)
        self.group = group
        self.simplifier = simplifier
        self.left = left
        self.right = right

    def __pow__(self, forms):
        paired = [self.group.pair_prod(self.left[i], self.right[i]) for i in range(self.n)]
        out = [1 for cl in range(forms.classes)]
        for i in range(self.n):
            batch = batch_exp(paired[i], forms.content[i], forms.nbits[i])
            for cl in range(forms.classes):
                out[cl] *= batch[cl]
        return out
