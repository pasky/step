"""
STEP ("Select the Easiest Point") is a scalar optimization algorithm
that minimizes a function by halving intervals over the bounded
space iteratively, each time selecting the interval with smallest
"difficulty".  The difficulty measure is curvature of x^2 function
crossing the interval boundary points and touching the supposed
(so-far-estimated) optimum; this curvature will be small for
intervals that have boundary points near the optimum.  The "smoother"
the function, the better this works.

http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=349896
http://www.applied-mathematics.net/optimization/Step.pdf

If you want to simply use STEP for straightforward scalar optimization,
you can invoke the ``step_minimize()`` function or invoke the algorithm
through ``scipy.optimize.minimize_scalar`` passing the return value
of ``step_minmethod()`` function as the method parameter.  Example:

>>> def f(x):
...     return (x - 2) * x * (x + 2)**2

>>> from step import step_minimize
>>> step_minimize(f, bounds=(-10, +10), maxiter=100)
{'fun': -9.91494958991847,
 'nit': 100,
 'success': True,
 'x': 1.2807846069335938}

>>> from step import step_minmethod
>>> import scipy.optimize as so
>>> so.minimize_scalar(f, bounds=(-10, +10), method=step_minmethod, options={'disp':False, 'maxiter':100})
     fun: -9.91494958991847
       x: 1.2807846069335938
 success: True
     nit: 100

You can also use the STEP class interface to single-step the algorithm,
possibly even tweaking its internal data structures between iterations.
We use that for multi-dimensional STEP.
"""

import math
import numpy as np
from operator import itemgetter


class STEP:
    """
    This class implements the scalar STEP algorithm run in a piece-meal
    way that allows simple scalar optimization as well as tweaking
    of internal STEP data within multidimensional wrappers.

    Example:

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    >>> import step
    >>> optimize = step.STEP(f)
    >>> optimize.begin(bounds=(-10,10))
    >>> for i in range(100):
    ...     (x, y) = optimize.one_step()
    ...     if y is None: break
    ...     if optimize.fmin < 1e-8: break
    >>> print(optimize.xmin, optimize.fmin)

    """
    def __init__(self, fun, epsilon=1e-8, disp=False, tolx=1e-11):
        """
        Set up a STEP algorithm instance on a particular function.
        This does not evaluate it in any way yet - to start optimization,
        call .begin(), then repeatedly .one_step().
        """
        self.fun = fun
        self.epsilon = epsilon
        self.disp = disp
        self.tolx = tolx

        # These will be filled in begin()
        self.points = None
        self.values = None
        self.xmin = None
        self.fmin = None
        self.difficulty = None
        self.axis = None

    def begin(self, bounds, point0=None, axis=None):
        """
        Initialize the algorithm with particular global interval bounds
        and starting point (the middle of the interval by default).

        If the bounds are in multi-dimensional space, axis denotes the
        axis along which scalar optimization is performed (with the
        other dimensions held fixed).
        """
        self.axis = axis

        if point0 is None:
            point0 = (bounds[0] + bounds[1]) / 2.0

        if axis is None:
            self.points = [bounds[0], point0, bounds[1]]
        else:
            self.points = [np.array(point0), point0, np.array(point0)]
            self.points[0][axis] = bounds[0][axis]
            self.points[2][axis] = bounds[1][axis]
        self.values = [self.fun(p) for p in self.points]

        print(self.points)
        print(self.values)
        imin, self.fmin = min(enumerate(self.values), key=itemgetter(1))
        self.xmin = self.points[imin]

        self._recompute_difficulty()

        return (self.xmin, self.fmin)

    def one_step(self):
        """
        Perform one iteration of the STEP algorithm, which amounts to
        selecting the interval to halve, evaluating the function once
        there and updating the interval difficulties.

        Returns the (x, y) tuple for the selected point (this is NOT
        the currently found optimum; grab that from .xmin, .fmin).
        Returns (None, None) if no step could have been performed
        anymore (this signals the algorithm should be terminated).
        """

        # Select the easiest interval which is wide enough
        def interval_wide_enough(i):
            if self.axis is None:
                delta = self.points[i+1] - self.points[i]
            else:
                delta = self.points[i+1][self.axis] - self.points[i][self.axis]
            return delta >= self.tolx
        idiff = filter(lambda (i, diff): interval_wide_enough(i),
                       enumerate(self.difficulty))
        if len(idiff) == 0:
            return (None, None)  # We cannot split the interval more
        i, diff = min(idiff, key=itemgetter(1))

        if self.disp:
            print('Easiest interval %f: [%f, %f]' % (diff, self.points[i], self.points[i+1]))

        # Split it into two
        newpoint = (self.points[i] + self.points[i+1]) / 2.0
        newvalue = self.fun(newpoint)
        self.points.insert(i+1, newpoint)
        self.values.insert(i+1, newvalue)
        self.difficulty[i] = None
        self.difficulty.insert(i+1, None)

        if newvalue < self.fmin:
            # New fmin, recompute difficulties of all intervals
            self.fmin = newvalue
            self.xmin = self.points[i+1]
            self._recompute_difficulty()
        else:
            # No fmin change, compute difficulties only of the two
            # new intervals
            self.difficulty[i] = self._interval_difficulty(self.points[i:i+2], self.values[i:i+2])
            self.difficulty[i+1] = self._interval_difficulty(self.points[i+1:i+3], self.values[i+1:i+3])

        return (newpoint, newvalue)

    def _interval_difficulty(self, points, values):
        """
        Compute difficulty of a single interval between two points.
        """
        # Recompute the second point coordinates with regards to the left (first)
        # point.
        if self.axis is None:
            x = points[1] - points[0]
        else:
            x = points[1][self.axis] - points[0][self.axis]
        y = values[1] - values[0]
        f = self.fmin - values[0] - self.epsilon

        # Curvature of parabole crossing [0,0], [x,y] and touching [?, f]
        a = (y - 2*f + 2*math.sqrt(f * (f - y))) / (x**2)
        return a

    def _recompute_difficulty(self):
        """
        Recompute the difficulty of all intervals.
        """
        difficulty = []
        for i in range(len(self.points) - 1):
            diff = self._interval_difficulty(self.points[i:i+2], self.values[i:i+2])
            difficulty.append(diff)
        self.difficulty = difficulty
        return difficulty


def step_minimize(fun, bounds, args=(), maxiter=100, callback=None, axis=None, point0=None, **options):
    """
    Minimize a given function within given bounds (a tuple of two points).

    The function can be multi-variate; in that case, you can pass numpy
    arrays as bounds, but you must also specify axis, as we still perform
    just scalar optimization along a specified axis.

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

    # Instantiate and fire off the STEP algorithm
    optimize = STEP(fun, **options)
    optimize.begin(bounds, point0=point0, axis=axis)

    niter = 0
    while niter < maxiter:
        (x, y) = optimize.one_step()
        if y is None:
            break

        if callback is not None:
            if callback(optimize.xmin):
                break

        niter += 1

    return dict(fun=optimize.fmin, x=optimize.xmin, nit=niter,
                success=(niter > 1))


def step_minmethod(fun, **options):
    """
    A scipy.optimize.minimize_scalar method callable to use for minimization
    within the SciPy optimization framework.

    Example:

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    >>> from step import step_minmethod
    >>> import scipy.optimize as so
    >>> so.minimize_scalar(f, bounds=(-10, +10), method=step_minmethod(), options={'disp':False, 'maxiter':100})
         fun: -9.91494958991847
           x: 1.2807846069335938
     success: True
         nit: 100

    """
    from scipy import optimize

    del options['bracket']

    result = step_minimize(fun, **options)
    return optimize.OptimizeResult(**result)
