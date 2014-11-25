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


def ndstep_minimize(fun, bounds, args=(), maxiter=2000, callback=None,
                    point0=None, dimselect=None, stagiter=None, **options):
    """
    Minimize a given multivariate function within given bounds
    (a tuple of two points).

    Each dimension is optimized by a separate STEP algorithm but the
    steps along different dimensions are interleaved and improving
    solutions along one dimensions are propagated to STEP intervals
    in all other dimensions.

    The stopping condition is either maxiter total iterations or when
    stagiter optimization steps are done without reaching an improvement
    (whichever comes first).  By default, stagiter is 10*DIM.

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
    every 10*DIM iterations; if it returns True, the optimization run is
    stopped.

    See the module description for an example.
    """

    dim = np.shape(bounds[0])[0]
    disp = options.get('disp', False)
    if stagiter is None:
        stagiter = 10 * dim
    callback_interval = 10 * dim

    xmin = None
    fmin = np.Inf

    optimize = [STEP(fun, **options) for i in range(dim)]
    for i in range(dim):
        (x, y) = optimize[i].begin(bounds, point0=point0, axis=i)

        if y < fmin:
            xmin = x
            fmin = y

    niter = -1
    niter_callback = callback_interval
    last_improvement = 0  # #iter that last brought some improvement
    while True:
        niter += 1

        # Test stopping conditions
        if maxiter is not None and niter >= maxiter:
            # Too many iterations
            break
        if last_improvement < niter - stagiter:
            # No improvement for the last #dim iterations
            break

        # Pick the next dimension to take a step in
        if dimselect is None:
            # By default, use round robin
            i = niter % dim
        else:
            i = dimselect(fun, optimize, niter, min=(xmin, fmin))

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
            last_improvement = niter

        if callback is not None and niter >= niter_callback:
            if callback(optimize[i].xmin):
                break
            niter_callback = niter + callback_interval

    return dict(fun=fmin, x=xmin, nit=niter,
                success=(niter > 1))

if __name__ == "__main__":
    """
    A simple testcase for speed benchmarking, etc.

    We optimize the Rastrigin-Bueche function in 20D in range [-5,5]
    for maxiter iterations, using ndstep_minimize() with random restarts.
    """
    maxiter = 16000

    # Reproducible runs
    np.random.seed(42)

    def f(xx):
        """ 20D Rastrigin-Bueche, with optimum displaced from zero to [-2,2] """
        x = xx - np.random.permutation(np.linspace(-2, 2, 20))
        return 10 * (20 - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
    x0 = np.zeros(20) - 5
    x1 = np.zeros(20) + 5

    globres = dict(fun=np.Inf, x=None, nit=0, restarts=0, success=False)
    while globres['fun'] > 1e-8 and globres['nit'] < maxiter:
        # Initial solution in a more interesting point than zero
        # to get rid of intrinsic regularities
        # When a minimization finishes, run a random restart then
        p0 = np.random.rand(20) * 2 - 1

        res = ndstep_minimize(f, bounds=(x0, x1), point0=p0, maxiter=(maxiter - globres['nit']))
        print('intermediate solution %s' % (res,))
        if res['fun'] < globres['fun']:
            globres['fun'] = res['fun']
            globres['x'] = res['x']
            globres['success'] = True
        globres['nit'] += res['nit']
        globres['restarts'] += 1

    print(globres)
