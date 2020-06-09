"""
Picks a random MNIST test set digit and runs encryption and Quadratic
evaluation on it.
"""

import random

import numpy as np
from core import discretelogarithm
from core import models
from core import scheme
from sklearn.datasets import fetch_openml

inst = 'objects/instantiations/MNT159.inst'
vector_length = 784
k = 250
classes = 10
model = 'objects/ml_models/final.mlm'
mnist = fetch_openml('mnist_784')
X, y = mnist['data'].astype('float'), mnist['target'].astype('float')
X_test, y_test = X[60000:], y[60000:]

index = random.randint(0, 9999)
X, y = X_test[index], y_test[index]

print(f'Will test on MNIST test instance #{index}, which is a {int(y)}.')
print('Importing model.')
ml = models.MLModel(source=model)
biased = np.ones(785)
biased[1:] = X
print('Done!\n')

results = ml.evaluate(biased)
print(f'Expected output:\n{results}\n')

print('Importing scheme.')

scheme = scheme.ML_DGP(inst)
print('Done!\n')

print('Loading discrete logarithm solver.')

dlog = discretelogarithm.PreCompBabyStepGiantStep(
    scheme.group, scheme.gt, minimum=-1.7e11, maximum=2.7e11, step=1 << 13,
)

scheme.set_dlog(dlog)
print('Done!\n')

print('Importing keys...')
pk = models.PublicKey(source=f'objects/pk/common_{vector_length}.pk')
msk = models.MasterKey(source=f'objects/msk/common_{vector_length}.msk')
print('Done!\n')

print('Encrypting...')
v = models.Vector(array=X)
c = scheme.encrypt(pk, v)
print('Done!\n')

print('Generating functional decryption key...')
dk = scheme.keygen(msk, ml)
print('Done!\n')

print('Decrypting...')
dec = scheme.decrypt(pk, dk, c)
print('Done!\n')

print(f'Decryption result:\n{dec}')
print(f'Image is believed to be a {np.argmax(dec)}.')

assert (dec == results.astype(int)).all(), 'Error, decryption failed'
