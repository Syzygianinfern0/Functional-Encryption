"""
A simple function that generates a pair of keys for a given length.
"""
from .scheme import ML_DGP


def make_keys(
    vector_length, inst, name='common', path='',
):
    scheme = ML_DGP(inst)
    pk, msk = scheme.setup(vector_length=vector_length)
    pk.toFile(f'{path}/pk', f'{name}_{vector_length}')
    msk.toFile(f'{path}/msk', f'{name}_{vector_length}')
