"""
STEP is a scalar optimization algorithm.  This module provides
an ``ndstep_minimize`` function that tries to apply it to multivariate
optimization nevertheless.

The approach is to find an optimum in each dimension by a separate STEP
algorithm but the steps along different dimensions are interleaved and
improving solutions along one dimensions are propagated to STEP
intervals in all other dimensions.

Example:

>>> def f(x):
...     return np.linalg.norm(x) ** 2
>>> x0 = np.array([-3, -3, -3])
>>> x1 = np.array([+1, +2, +3])

>>> from ndstep import ndstep_minimize
>>> ndstep_minimize(f, bounds=(x0, x1), maxiter=1000)
{'fun': 3.637978807091713e-12,
 'nit': 1000,
 'success': True,
 'x': array([  0.00000000e+00,   1.90734863e-06,   0.00000000e+00])}

"""


import numpy as np

from step import STEP


def ndstep_minimize(fun, bounds, args=(), maxiter=100, callback=None,
                    point0=None, dimselect=None, **options):
    """
    Minimize a given multivariate function within given bounds
    (a tuple of two points).

    Each dimension is optimized by a separate STEP algorithm but the
    steps along different dimensions are interleaved and improving
    solutions along one dimensions are propagated to STEP intervals
    in all other dimensions.

    Dimensions are selected using a round-robin strategy by default.
    You can pass a custom dimension selection function that is called
    as dimselect(fun, [step...], niter, min=(xmin, fmin)):

    >>> # Rastrigin-Bueche
    >>> def f(x): return 10 * (20 - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
    >>> x0 = np.ones(20) - 5
    >>> x1 = np.ones(20) + 5
    # Random dimension choice
    >>> ndstep_minimize(f, bounds=(x0, x1), maxiter=2000,
    ...     dimselect=lambda fun, optimize, niter, min:
    ...         np.random.permutation(range(len(optimize)))[0])
    # Easiest dimensions choice
    >>> p0 = np.random.rand(20)
    >>> ndstep_minimize(f, bounds=(x0, x1), point0=p0, maxiter=2000,
    ...     dimselect=lambda fun, optimize, niter, min:
    ...         np.argmin([o.difficulty[o.easiest_interval()] for o in optimize])
    ...             if niter >= len(optimize) * 4
    ...             else niter % len(optimize))

    The callback, if passed, is called with the current optimum hypothesis
    every 100 iterations; if it returns True, the optimization run is
    stopped.

    See the module description for an example.
    """

    dim = np.shape(bounds[0])[0]
    disp = options.get('disp', False)
    callback_interval = 100

    xmin = None
    fmin = np.Inf

    optimize = [STEP(fun, **options) for i in range(dim)]
    for i in range(dim):
        (x, y) = optimize[i].begin(bounds, point0=point0, axis=i)

        if y < fmin:
            xmin = x
            fmin = y

    niter = 0
    niter_callback = callback_interval
    while niter < maxiter:
        # Pick the next dimension to take a step in
        if dimselect is None:
            # By default, use round robin
            i = niter % dim
        else:
            i = dimselect(fun, optimize, niter, min=(xmin, fmin))

        niter += 1

        if optimize[i] is None:
            continue

        if disp: print('-----------------------', i)
        x0 = optimize[i].xmin
        y0 = optimize[i].fmin
        (x, y) = optimize[i].one_step()
        if disp: print(x, y)
        if y is None:
            optimize[i] = None
            continue

        if y < y0:
            # We found an improving solution, shift the "context" on
            # all other axes
            if disp: print('improving solution!')
            xmin = x
            fmin = y
            for j in range(dim):
                if i == j:
                    continue
                optimize[j].update_context(x - x0, y - y0)

        if callback is not None and niter >= niter_callback:
            if callback(optimize[i].xmin):
                break
            niter_callback = niter + callback_interval

    return dict(fun=fmin, x=xmin, nit=niter,
                success=(niter > 1))

if __name__ == "__main__":
    """
    A simple testcase for speed benchmarking, etc.
    """
    # Reproducible runs
    np.random.seed(42)

    def f(xx):
        """ 20D Rastrigin-Bueche, shifted from zero in most dimensions """
        x = xx + np.array([0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1])-1
        return 10 * (20 - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
    x0 = np.zeros(20) - 5
    x1 = np.zeros(20) + 5

    # Initial solution in a more interesting point than zero
    # to get rid of intrinsic regularities
    p0 = np.random.rand(20)

    res = ndstep_minimize(f, bounds=(x0, x1), point0=p0, maxiter=2000)
    print(res)
