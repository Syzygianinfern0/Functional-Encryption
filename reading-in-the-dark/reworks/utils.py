import numpy as np


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
