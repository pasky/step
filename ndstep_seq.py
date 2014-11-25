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


def ndstep_seq_minimize(fun, bounds, args=(), maxiter=2000, maxiter_uni=100,
                        callback=None, point0=None, dimselect=None,
                        **options):
    """
    Minimize a given multivariate function within given bounds
    (a tuple of two points).

    Sequentially optimize along each axis separately, each for
    maxiter_uni iterations.  The stopping condition is either
    maxiter total iterations or when one round of optimizations
    is done without reaching an improvement (whichever comes first).

    Dimensions are selected using a round-robin strategy by default.
    You can pass a custom dimension selection function that is called
    as dimselect(fun, dim, niter_inner, niter_outer, min=(xmin, fmin)):

    >>> ndstep_seq_minimize(f, bounds=(x0, x1), maxiter_uni=5,
    ...     dimselect=lambda fun, dim, niter_inner, niter_outer, min:
    ...         np.random.permutation(range(dim))[0])


    See the module description for an example.
    """

    dim = np.shape(bounds[0])[0]
    disp = options.get('disp', False)

    xmin = None
    fmin = np.Inf

    niter_inner = 0  # total number of STEP iterations (across all dimensions)
    niter_outer = 0  # total number of dimension iterations
    last_improvement = 0  # #iter_outer that last brought some improvement
    while True:
        # Test stopping conditions
        if maxiter is not None and niter_inner >= maxiter:
            # Too many iterations
            break
        if last_improvement < niter_outer - dim:
            # No improvement for the last #dim iterations
            break

        # Select axis
        if dimselect is None:
            # By default, in simple round-robin fashion
            axis = niter_outer % dim
        else:
            axis = dimselect(fun, dim, niter_inner, niter_outer, min=(xmin, fmin))

        if disp: print('---------------- %d' % (niter_outer % dim))
        res = step_minimize(fun, bounds=bounds, point0=xmin, maxiter=maxiter_uni,
                            axis=axis)
        if disp: print('===>', res['x'], res['fun'])

        if res['fun'] < fmin:
            if disp: print('improving!')
            fmin = res['fun']
            xmin = res['x']
            last_improvement = niter_outer
        niter_inner += res['nit']
        niter_outer += 1

    return dict(fun=fmin, x=xmin, nit=niter_inner, success=(niter_inner > 1))
