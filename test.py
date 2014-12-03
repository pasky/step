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
#
# If you copy or symlink the bbobbenchmarks.py file from the BBOB COCO
# benchmark suite to the current directory, you can benchmark any of
# these functions too by specifying e.g. -f b7 for function 7 (rotated
# ellipsoid).

from __future__ import print_function

import getopt
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


class F4:
    """ Rastrigin-Bueche """
    def __init__(self, dim):
        self.dim = dim
        self.optimum = np.random.permutation(np.linspace(-4, 4, self.dim))

    def opt_y(self):
        return 0

    def __call__(self, xx):
        x = xx - self.optimum
        return 10 * (self.dim - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)


class BBOB:
    """ A BBOB function """
    def __init__(self, dim, fid, iid):
        import bbobbenchmarks
        self.dim = dim
        (self.f, self.fopt) = bbobbenchmarks.instantiate(fid, iinstance=iid)

    def opt_y(self):
        self.optimum = self.f.xopt
        return self.fopt

    def __call__(self, x):
        return self.f(x)


class BBOBFactory:
    """ A BBOB function factory """
    def __init__(self, fid, iid=1):
        self.fid = fid
        self.iid = iid

    def __call__(self, dim):
        return BBOB(dim, self.fid, self.iid)


def dimselect_random(fun, optimize, niter, min):
    return np.random.randint(len(optimize))


def dimselect_mindiff(fun, optimize, niter, min):
    if niter >= len(optimize) * 4:
        return np.argmin([o.difficulty[o.easiest_interval()] for o in optimize])
    else:
        return niter % len(optimize)


def dimselect_maxdiff(fun, optimize, niter, min):
    if niter >= len(optimize) * 4:
        return np.argmax([o.difficulty[o.easiest_interval()] for o in optimize])
    else:
        return niter % len(optimize)


def dimselect_diffpd(fun, optimize, niter, min):
    if niter >= len(optimize) * 4:
        pd = np.array([o.difficulty[o.easiest_interval()] for o in optimize])
        pd /= np.sum(pd)
        return np.random.choice(range(len(optimize)), p=pd)
    else:
        return niter % len(optimize)


def dimselect_rdiffpd(fun, optimize, niter, min):
    if niter >= len(optimize) * 4:
        pd = np.array([o.difficulty[o.easiest_interval()] for o in optimize])
        pd = 1. / pd
        pd /= np.sum(pd)
        return np.random.choice(range(len(optimize)), p=pd)
    else:
        return niter % len(optimize)


def run_ndstep(logfname, minimize_function, options):
    """
    A simple testcase for speed benchmarking, etc.

    We optimize the Rastrigin-Bueche function in 20D in range [-5,5]
    for maxiter iterations, using ndstep_minimize() with random restarts.
    """
    # Reproducible runs
    np.random.seed(options['seed'])

    dim = options['dim']
    f = options['f'](dim)
    logf = open(logfname, mode='w')

    x0 = np.zeros(dim) - 5
    x1 = np.zeros(dim) + 5

    globres = dict(fun=np.Inf, x=None, nit=0, restarts=0, success=False)
    while globres['fun'] > 1e-8 and globres['nit'] < options['maxiter']:
        # Initial solution in a more interesting point than zero
        # to get rid of intrinsic regularities
        # When a minimization finishes, run a random restart then
        p0 = np.random.rand(dim) * 4 - 1

        res = minimize_function(lambda x: f(x),
                                bounds=(x0, x1), point0=p0,
                                maxiter=(options['maxiter'] - globres['nit']),
                                callback=lambda x, y: y - f.opt_y() <= 1e-8,
                                logf=logf, dimselect=options['dimselect'])
        res['fun'] -= f.opt_y()
        print(_format_solution(res, f.optimum))
        if res['fun'] < globres['fun']:
            globres['fun'] = res['fun']
            globres['x'] = res['x']
            globres['success'] = True
        globres['nit'] += res['nit']
        globres['restarts'] += 1

    print(globres)
    print(_format_solution(globres, f.optimum))


def usage(err=2):
    print('Benchmark ndstep, ndstep_seq')
    print('Usage: test.py [-f {f4,bFID}] [-d DIM] [-e {rr,random,mindiff,maxdiff,diffpd,rdiffpd}] [-i MAXITER] [-s SEED] {ndstep,ndstep_seq}')
    sys.exit(err)


if __name__ == "__main__":
    # Deal with options and such

    options = {
        'f': F4,
        'dim': 20,
        'maxiter': 32000,
        'seed': 43,
        'dimselect': None,
    }

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:e:f:hi:s:", ["help"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()

    for o, a in opts:
        if o in ("-h", "--help"):
            usage(0)
        elif o == "-e":
            dimstrats = dict(rr=None, random=dimselect_random,
                             mindiff=dimselect_mindiff, maxdiff=dimselect_maxdiff,
                             diffpd=dimselect_diffpd, rdiffpd=dimselect_rdiffpd)
            options['dimselect'] = dimstrats[a]
        elif o == "-f":
            if a == "f4":
                options['f'] = F4
            elif a.startswith('b'):
                options['f'] = BBOBFactory(int(a[1:]))
            else:
                usage()
        elif o == "-d":
            options['dim'] = int(a)
        elif o == "-i":
            options['maxiter'] = int(a)
        elif o == "-s":
            options['seed'] = int(a)
        else:
            assert False, "unhandled option"

    method = args[0]

    # Now, actually run the circus!

    if method == "ndstep":
        run_ndstep('ndstep-log.txt', ndstep_minimize, options)
    elif method == "ndstep_seq":
        run_ndstep('ndstep_seq-log.txt', ndstep_seq_minimize, options)
    else:
        assert False
