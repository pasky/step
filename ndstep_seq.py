import numpy as np

from step import step_minimize


def ndstep_seq_minimize(fun, bounds, args=(), maxiter=None, maxiter_uni=100, callback=None, point0=None, **options):
    """
    Minimize a given multivariate function within given bounds
    (a tuple of two points).

    Sequentially optimize along each axis separately, each for
    maxiter_uni iterations.

    Example:

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    >>> from step import step_minimize
    >>> step_minimize(f, bounds=(-10, +10), maxiter=100)
    {'fun': -9.91494958991847,
     'nit': 100,
     'success': True,
     'x': 1.2807846069335938}

    """

    dim = np.shape(bounds[0])[0]
    if maxiter is None:
        maxiter = maxiter_uni * dim
    try:
        disp = options.get('disp', False)
    except KeyError:
        disp = False

    xmin = None
    fmin = np.Inf

    axis = 0
    niter = 0
    while niter < maxiter:
        if disp: print('---------------- %d' % (axis % dim))
        res = step_minimize(fun, bounds=bounds, point0=xmin, maxiter=maxiter_uni, axis=(axis % dim))
        if disp: print('===>', res['x'], res['fun'])

        if res['fun'] < fmin:
            if disp: print('improving!')
            fmin = res['fun']
            xmin = res['x']
        niter += res['nit']
        axis += 1

    return dict(fun=fmin, x=xmin, nit=niter, success=(niter > 1))
