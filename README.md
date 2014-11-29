Python Implementation of the STEP Optimization Algorithm
========================================================

STEP ("Select the Easiest Point") is a scalar optimization algorithm
that minimizes a function by halving intervals over the bounded
space iteratively, each time selecting the interval with smallest
"difficulty".  The difficulty measure is curvature of x^2 function
crossing the interval boundary points and touching the supposed
(so-far-estimated) optimum; this curvature will be small for
intervals that have boundary points near the optimum.  The "smoother"
the function, the better this works.

  * http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=349896
  * http://www.applied-mathematics.net/optimization/Step.pdf

For now, see the top of the ``step.py`` module for usage instructions
and some simple examples.

Experimental support for multi-variate optimization of linearly
separable functions is provided by ``ndstep.py`` and ``ndstep-seq.py``.

A testing tool for multi-variate optimization using STEP is available
as the ``test.py`` script.
