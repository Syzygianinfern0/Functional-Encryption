"""
This should be the first file you run.

Generates your master secret key and the corresponding public key, and
fills the database with precomputations.

"""

import os

from core import discretelogarithm
from core import make_keys
from core import scheme

inst = 'objects/instantiations/MNT159.inst'
model = 'objects/ml_models/final.mlm'
vector_length = 784

if not os.path.exists('objects/msk'):
    os.makedirs('objects/msk')
if not os.path.exists('objects/pk'):
    os.makedirs('objects/pk')
if not os.path.exists(f'objects/msk/common_{vector_length}.msk'):
    print('Generating keys.')
    make_keys.make_keys(
        vector_length, inst=inst, name='common', path='objects',
    )
    print('Done!\n')
else:
    print('Keys were already generated.\n')
print('Precomputing discrete logarithm.')
scheme = scheme.ML_DGP(inst)
dlog = discretelogarithm.PreCompBabyStepGiantStep(
    scheme.group, scheme.gt, minimum=-1.7e11, maximum=2.7e11, step=1 << 13,
)
dlog.precomp()
print('Done!\n')
