import collections

import numpy as np


def is_array(a):
    return isinstance(a, (collections.Sequence, np.ndarray))


def is_scalar(s):
    return (
        isinstance(s, int) or isinstance(s, np.int64) or (isinstance(s, float) and s.is_integer())
    )


def exp(g, x):
    """
    Redefine exponentiation to avoid.

    ```pairing.Error: undefined exponentiation operation.```

    on non unitX negative integers

    """
    return (g ** abs(x)) ** np.sign(x)


def fast_exp_const_time(g, X, xmin=0, xmax=255):
    """
    Precompute every exponentiation between xmin and xmax once and for all,
    then use the results to compute the exponentiations in X without
    introducing obvious timing issues.
    """
    curr = exp(g, xmin)
    precomp = [curr]
    for i in range(xmax - xmin):
        curr *= g
        precomp.append(curr)
    return [precomp[x - xmin] for x in X]


def batch_exp(g, X, nbits):
    """
    Compute the g^X where X are ints of at most nbits bits.

    NOT CONSTANT TIME

    """
    one = g ** 0
    powers = [g]
    curr = g
    for b in range(nbits - 1):
        curr = curr * curr
        powers.append(curr)
    out = []
    for x in X:
        bit = 1
        curr = one
        val = abs(x)
        for b in range(nbits):
            if val & bit:
                curr *= powers[b]
            bit *= 2
        if x < 0:
            curr = 1 / curr
        out.append(curr)
    return out
