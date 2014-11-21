"""
STEP is a scalar optimization algorithm.  This module provides
an ``ndstep_seq_minimize`` function that tries to apply it to
multivariate optimization nevertheless.

The approach is to simply keep running independent STEP algorithms
along separate dimensions, progressively finding improved solutions.
We try to mirror the approach described in (Posik, 2009):

  * http://dl.acm.org/citation.cfm?id=1570325
  * http://sci2s.ugr.es/EAMHCO/pdfs/contributionsGECCO09/p2329-posik.pdf

Example:

>>> def f(x):
...     return np.linalg.norm(x) ** 2
>>> x0 = np.array([-3, -3, -3])
>>> x1 = np.array([+1, +2, +3])

>>> from ndstep import ndstep_minimize
>>> ndstep_minimize_seq(f, bounds=(x0, x1), maxiter_uni=100)
{'fun': 5.8207660913467407e-11,
 'nit': 300,
 'success': True,
 'x': array([  0.00000000e+00,  -7.62939453e-06,   0.00000000e+00])}

"""


import numpy as np

from step import step_minimize


def ndstep_seq_minimize(fun, bounds, args=(), maxiter=None, maxiter_uni=100, callback=None, point0=None, **options):
    """
    Minimize a given multivariate function within given bounds
    (a tuple of two points).

    Sequentially optimize along each axis separately, each for
    maxiter_uni iterations.

    Dimensions are selected using a round-robin strategy.

    See the module description for an example.
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
