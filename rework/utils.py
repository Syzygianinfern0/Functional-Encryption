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


# Compute the g^X where X are ints of at most nbits bits.
# NOT CONSTANT TIME
def batch_exp(g, X, nbits):
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
