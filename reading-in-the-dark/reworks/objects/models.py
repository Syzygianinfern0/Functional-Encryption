import json
import os
import zlib
from base64 import b64decode
from base64 import b64encode

import numpy as np
from charm.core.engine.util import deserializeObject
from charm.core.engine.util import from_json
from charm.core.engine.util import serializeObject
from charm.core.engine.util import to_json
from charm.toolbox.pairinggroup import PairingGroup
from exceptions import WrongInputError
from objects.poly_nets import DiagonalQuadraticForms
from objects.poly_nets import Projection
from utils import is_array
from vectors.enc_vector import EncryptedVector
from vectors.keys import MasterKey


class Serializable:
    """
    Base class handles some of the trouble of saving objects to a file.
    """

    ext_ = 'ser'

    def serialize(self):
        d = self.__dict__.copy()
        if 'group' in d:
            del d['group']
            return {'param': self.group.param, 'rest': serializeObject(d, self.group)}
        else:
            return d

    def deserialize(self, o):
        assert type(o) == dict, 'Wrong file format.'
        if 'param' in o:
            self.group = PairingGroup(o['param'])
            assert 'rest' in o, 'Wrong file format.'
            d = deserializeObject(o['rest'], self.group)
        else:
            d = o
        for key, value in d.items():
            setattr(self, key, value)
        return self

    def toFile(self, path, title):
        serialized = self.serialize()
        js = json.dumps(serialized, default=to_json)
        by = bytes(js, 'utf-8')
        compressed = zlib.compress(by)
        encoded = b64encode(compressed)
        final = encoded.decode('utf-8')
        with open(os.path.join(path, f'{title}.{self.ext_}',), 'w',) as f:
            f.write(final)

    def fromFile(self, source):
        assert isinstance(source, str)
        assert os.path.isfile(source)
        with open(source, 'r') as f:
            final = f.read()
            encoded = bytes(final, 'utf-8')
            compressed = b64decode(encoded)
            by = zlib.decompress(compressed)
            js = by.decode('utf-8')
            serialized = json.loads(js, object_hook=from_json)
            return self.deserialize(serialized)


class MLModel(Serializable):
    ext_ = 'mlm'

    def __init__(self, proj=None, forms=None, source=''):
        if source:
            self.fromFile(source)
            return
        assert proj
        assert forms
        assert isinstance(proj, Projection)
        assert isinstance(forms, DiagonalQuadraticForms)
        assert forms.k == proj.k
        self.proj = proj
        self.forms = forms

    def evaluate(self, X):
        if isinstance(X, EncryptedVector):
            return (self.proj * X) ** self.forms
        elif isinstance(X, MasterKey):
            Ps = self.proj * X.s
            Pt = self.proj * X.t
            element_wise_prod = np.multiply(Ps, Pt)
            return np.dot(np.transpose(self.forms.content), element_wise_prod).tolist()
        elif is_array(X):
            PXsquared = np.square(self.proj * X)
            return np.dot(np.transpose(self.forms.content), PXsquared)
        else:
            raise WrongInputError(X)

    def serialize(self):
        proj = self.proj
        forms = self.forms
        del self.proj
        del self.forms
        self.proj_ = proj.serialize()
        self.forms_ = forms.serialize()
        return super(MLModel, self).serialize()

    def deserialize(self, o):
        super(MLModel, self).deserialize(o)
        self.proj = Projection().deserialize(self.proj_)
        del self.proj_
        self.forms = DiagonalQuadraticForms().deserialize(self.forms_)
        del self.forms_
        return self

    def naive_bounds(self):
        maxes = []
        P = np.transpose(self.proj.columns)
        for j in range(self.proj.k):
            neg = sum([255 * p if p < 0 else 0 for p in P[j][1:]]) + P[j][0]
            pos = sum([255 * p if p > 0 else 0 for p in P[j][1:]]) + P[j][0]
            maxes.append(max(-neg, pos) ** 2)
        largest = 0
        lowest = 0
        matrix = np.transpose(self.forms.content)
        for cl in range(self.forms.classes):
            neg = sum(
                [
                    (-maxes[j] * matrix[cl][j] if matrix[cl][j] < 0 else 0)
                    for j in range(self.proj.k)
                ]
            )
            pos = sum(
                [(maxes[j] * matrix[cl][j] if matrix[cl][j] > 0 else 0) for j in range(self.proj.k)]
            )
            if neg > lowest:
                lowest = neg
            if pos > largest:
                largest = pos
        return -lowest, largest

    def get_accuracy(self):
        assert self.proj.n == 785, 'get_accuracy only works for mnist'
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets('/tmp/data/')
        X_test_ = np.round(255 * mnist.test.images)
        y_test = mnist.test.labels.astype('int')
        X_test = np.ones((10000, 785))
        X_test[:, 1:] = X_test_
        good = 0
        for i in range(10000):
            good += 1 if (np.argmax(self.evaluate(X_test[i])) == y_test[i]) else 0
        print(f'Accuracy: {good / 100}%')
