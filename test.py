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


def run_ndstep():
    """
    A simple testcase for speed benchmarking, etc.

    We optimize the Rastrigin-Bueche function in 20D in range [-5,5]
    for maxiter iterations, using ndstep_minimize() with random restarts.
    """
    maxiter = 32000
    logf = open('ndstep-log.txt', mode='w')

    # Reproducible runs
    np.random.seed(42)

    optimum = np.random.permutation(np.linspace(-2, 2, 20))
    def f(xx):
        """ 20D Rastrigin-Bueche, with optimum displaced from zero to [-2,2] """
        x = xx - optimum
        return 10 * (20 - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
    x0 = np.zeros(20) - 5
    x1 = np.zeros(20) + 5

    globres = dict(fun=np.Inf, x=None, nit=0, restarts=0, success=False)
    while globres['fun'] > 1e-8 and globres['nit'] < maxiter:
        # Initial solution in a more interesting point than zero
        # to get rid of intrinsic regularities
        # When a minimization finishes, run a random restart then
        p0 = np.random.rand(20) * 2 - 1

        res = ndstep_minimize(f, bounds=(x0, x1), point0=p0,
                              maxiter=(maxiter - globres['nit']),
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


def run_ndstep_seq():
    """
    A simple testcase for speed benchmarking, etc.

    We optimize the Rastrigin-Bueche function in 20D in range [-5,5]
    for maxiter iterations, using ndstep_minimize() with random restarts.
    """
    maxiter = 32000
    logf = open('ndstep_seq-log.txt', mode='w')

    # Reproducible runs
    np.random.seed(42)

    optimum = np.random.permutation(np.linspace(-2, 2, 20))
    def f(xx):
        """ 20D Rastrigin-Bueche, with optimum displaced from zero to [-2,2] """
        x = xx - optimum
        return 10 * (20 - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
    x0 = np.zeros(20) - 5
    x1 = np.zeros(20) + 5

    globres = dict(fun=np.Inf, x=None, nit=0, restarts=0, success=False)
    while globres['fun'] > 1e-8 and globres['nit'] < maxiter:
        # Initial solution in a more interesting point than zero
        # to get rid of intrinsic regularities
        # When a minimization finishes, run a random restart then
        p0 = np.random.rand(20) * 2 - 1

        res = ndstep_seq_minimize(f, bounds=(x0, x1), point0=p0,
                              maxiter=(maxiter - globres['nit']),
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
    if sys.argv[1] == "ndstep":
        run_ndstep()
    elif sys.argv[1] == "ndstep_seq":
        run_ndstep_seq()
    else:
        assert False
