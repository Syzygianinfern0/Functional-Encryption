"""
Runs encryption and decryption on the entire MNIST test set, and reports
timings.
"""

import timeit

import numpy as np
from charm.toolbox.pairinggroup import pair
from core import discretelogarithm
from core import models
from core import scheme
from sklearn.datasets import fetch_openml

# assert False, 'Remember to run with -O for optimized results.'

inst = 'objects/instantiations/MNT159.inst'
vector_length = 784
model = 'objects/ml_models/final.mlm'
mnist = fetch_openml('mnist_784')
X_full, y_full = mnist['data'].astype('float'), mnist['target'].astype('float')
X_test, y_test = X_full[60000:], y_full[60000:]

# Randomly shuffle test set (with a set seed)
np.random.seed(42)
sigma = np.random.permutation(len(X_test))
X_test = X_test[sigma]
y_test = y_test[sigma]

print('Importing scheme.')

scheme = scheme.ML_DGP(inst)
print('Done!\n')

print('Importing model.')
ml = models.MLModel(source=model)
print('Done!\n')

print('Importing discrete logarithm.')

dlog = discretelogarithm.PreCompBabyStepGiantStep(
    scheme.group, scheme.gt, minimum=-1.7e11, maximum=2.7e11, step=1 << 13,
)

scheme.set_dlog(dlog)
print('Done!\n')

print('Importing keys...')
pk = models.PublicKey(source=f'objects/pk/common_{vector_length}.pk')
msk = models.MasterKey(source=f'objects/msk/common_{vector_length}.msk')
print('Done!\n')

biased = np.ones(785)

print('Generating functional decryption key...')
dk = scheme.keygen(msk, ml)
print('Done!\n')

enc_total = 0
eval_total = 0
dlog_total = 0
correct = 0
errors = 0
total = 0

for i in range(len(X_test)):
    biased[1:] = X_test[i]
    results = ml.evaluate(biased)

    v = models.Vector(array=X_test[i])
    before_encrypt = timeit.default_timer()
    c = scheme.encrypt(pk, v)
    after_encrypt = timeit.default_timer()
    enc_time = after_encrypt - before_encrypt
    before_eval = timeit.default_timer()
    evaluation = dk.model.evaluate(c)
    decrypted = [evaluation[i] * pair(c.simplifier, dk.skf[i]) for i in range(dk.classes)]
    after_eval = timeit.default_timer()
    eval_time = after_eval - before_eval
    before_dlog = timeit.default_timer()
    dec = list(map(dlog.solve, decrypted))
    after_dlog = timeit.default_timer()
    dlog_time = after_dlog - before_dlog

    if (x is not None for x in dec):
        if (dec == results.astype(int)).all():
            enc_total += enc_time
            eval_total += eval_time
            dlog_total += dlog_time
            total += 1
            if np.argmax(dec) == y_test[i]:
                correct += 1
        else:
            print('!' * 1000)
            print('Error, decryption failed')
            errors += 1
    else:
        print('!' * 1000)
        print('Error! Dlog failed!')
        errors += 1
    if total % 5 == 0:
        print('\n\n')
        print('-' * 40)
        print('Running stats:')
        print(f'Total error-free samples: {total}')
        print(f'Average encryption time: {enc_total / total}s.')
        print(f'Average evaluation time: {eval_total / total}s.')
        print(f'Average dlog time: {dlog_total / total}s.')
        print(f'Average accuracy: {100 * correct / total}.')
        print(f'Total errors: {errors}.')
        print('-' * 40)
        print('\n\n')
