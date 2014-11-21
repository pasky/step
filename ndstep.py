import numpy as np

from step import STEP


def ndstep_minimize(fun, bounds, args=(), maxiter=100, callback=None, point0=None, **options):
    """
    Minimize a given function within given bounds (a tuple of two points).

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
    try:
        disp = options.get('disp', False)
    except KeyError:
        disp = False

    xmin = None
    fmin = np.Inf

    optimize = [STEP(fun, **options) for i in range(dim)]
    for i in range(dim):
        (x, y) = optimize[i].begin(bounds, point0=point0, axis=i)

        if y < fmin:
            xmin = x
            fmin = y

    niter = 0
    while niter < maxiter:
        i = niter % dim
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

        if callback is not None:
            if callback(optimize[i].xmin):
                break

    return dict(fun=fmin, x=xmin, nit=niter,
                success=(niter > 1))
