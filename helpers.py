# Miscellaneous helper functions
from pydrake.autodiffutils import AutoDiffXd
import numpy as np

def jacobian2(function, x):
    """
    This is a rewritting of the jacobian function from drake which addresses
    a strange bug that prevents computations of Jdot.

    Compute the jacobian of the function evaluated at the vector input x
    using Eigen's automatic differentiation. The dimension of the jacobian will
    be one more than the output of ``function``.

    ``function`` should be vector-input, and can be any dimension output, and
    must return an array with AutoDiffXd elements.
    """
    x = np.asarray(x)
    assert x.ndim == 1, "x must be a vector"
    x_ad = np.empty(x.shape, dtype=np.object)
    for i in range(x.size):
        der = np.zeros(x.size)
        der[i] = 1
        x_ad.flat[i] = AutoDiffXd(x.flat[i], der)
    y_ad = np.asarray(function(x_ad))

    yds = []
    for y in y_ad.flat:
        yd = y.derivatives()
        if yd.shape == (0,):
            yd = np.zeros(x_ad.shape)
        yds.append(yd)

    return np.vstack(yds).reshape(y_ad.shape + (-1,))
