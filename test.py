#!/usr/bin/python
#
# A simple testcase runner for speed benchmarking, etc.
#
# We optimize the Rastrigin-Bueche function in 20D in range [-5,5]
# for maxiter iterations, using ndstep_minimize() or ndstep_seq_minimize()
# with random restarts.
#
# Usage: ./test.py ndstep
# Usage: ./test.py ndstep_seq

from __future__ import print_function

import numpy as np
import sys

from ndstep import ndstep_minimize
from ndstep_seq import ndstep_seq_minimize


def _format_solution(res, optimum):
    """
    Return a string describing the solution described in res,
    relative to the optimum point.
    """
    delta = np.abs(res['x'] - optimum)
    closest_d = np.min(delta)
    farthest_d = np.max(delta)
    avg_d = np.average(delta)
    sd_d = np.std(delta)
    distance = np.linalg.norm(delta)
    solstr = 'y=%e  nit=% 6d  dx=(min=%e, max=%e, avg=%.3f (+- %.3f = %.3f), dist=%e)' % \
             (res['fun'], res['nit'],
              closest_d, farthest_d, avg_d, sd_d, avg_d + sd_d, distance)
    return solstr


def f4(dim, optimum, xx):
    """ Rastrigin-Bueche """
    x = xx - optimum
    return 10 * (dim - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)


def run_ndstep(logfname, minimize_function, options):
    """
    A simple testcase for speed benchmarking, etc.

    We optimize the Rastrigin-Bueche function in 20D in range [-5,5]
    for maxiter iterations, using ndstep_minimize() with random restarts.
    """
    dim = options['dim']
    f = options['f']
    logf = open(logfname, mode='w')

    # Reproducible runs
    np.random.seed(options['seed'])

    optimum = np.random.permutation(np.linspace(-4, 4, dim))
    x0 = np.zeros(dim) - 5
    x1 = np.zeros(dim) + 5

    globres = dict(fun=np.Inf, x=None, nit=0, restarts=0, success=False)
    while globres['fun'] > 1e-8 and globres['nit'] < options['maxiter']:
        # Initial solution in a more interesting point than zero
        # to get rid of intrinsic regularities
        # When a minimization finishes, run a random restart then
        p0 = np.random.rand(dim) * 4 - 1

        res = minimize_function(lambda x: f(dim, optimum, x),
                                bounds=(x0, x1), point0=p0,
                                maxiter=(options['maxiter'] - globres['nit']),
                                callback=lambda x, y: y <= 1e-8,
                                logf=logf)
        print(_format_solution(res, optimum))
        if res['fun'] < globres['fun']:
            globres['fun'] = res['fun']
            globres['x'] = res['x']
            globres['success'] = True
        globres['nit'] += res['nit']
        globres['restarts'] += 1

    print(globres)
    print(_format_solution(globres, optimum))


if __name__ == "__main__":
    method = sys.argv[1]

    options = {
        'f': f4,
        'dim': 20,
        'maxiter': 32000,
        'seed': 43,
    }

    if method == "ndstep":
        run_ndstep('ndstep-log.txt', ndstep_minimize, options)
    elif method == "ndstep_seq":
        run_ndstep('ndstep_seq-log.txt', ndstep_seq_minimize, options)
    else:
        assert False
