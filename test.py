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
#
# Use -r N to repeat the measurements N times (with consecutive seeds)
# and show the averages.

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
    return np.argmin([o.difficulty[o.easiest_interval()] for o in optimize])


def dimselect_maxdiff(fun, optimize, niter, min):
    return np.argmax([o.difficulty[o.easiest_interval()] for o in optimize])


def dimselect_diffpd(fun, optimize, niter, min):
    pd = np.array([o.difficulty[o.easiest_interval()] for o in optimize])
    pd /= np.sum(pd)
    return np.random.choice(range(len(optimize)), p=pd)


def dimselect_rdiffpd(fun, optimize, niter, min):
    pd = np.array([o.difficulty[o.easiest_interval()] for o in optimize])
    pd = 1. / pd
    pd /= np.sum(pd)
    return np.random.choice(range(len(optimize)), p=pd)


class DimSelectHistory:
    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def update(self, min):
        (xmin, fmin) = min
        if self.lastdim >= 0:
            if fmin < self.lastfmin:
                self.hist[self.lastdim].append(self.lastfmin - fmin)
            else:
                self.hist[self.lastdim].append(0)
        if fmin < self.lastfmin:
            self.lastfim = fmin

    def __call__(self, fun, optimize, niter, min):
        return np.argmax([np.mean(self.hist[i]) for i in range(len(self.hist))])

    def reset(self):
        self.hist = [[] for i in range(self.dim)]
        self.lastdim = -1
        self.lastfmin = 1e10


class DimSelectHistoryRA:
    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def update(self, min):
        # Record results of previous selection
        (xmin, fmin) = min
        if self.lastdim >= 0:
            if fmin < self.lastfmin:
                delta = self.lastfmin - fmin
            else:
                delta = 0
            if self.runmean[self.lastdim] is None:
                self.runmean[self.lastdim] = delta
            else:
                beta = 1/10  # 1/beta should be < stagiter
                self.runmean[self.lastdim] = beta * delta + (1 - beta) * self.runmean[self.lastdim]
        if fmin < self.lastfmin:
            self.lastfim = fmin

    def __call__(self, fun, optimize, niter, min):
        # New selection
        return np.argmax([self.runmean[i] for i in range(len(self.runmean))])

    def reset(self):
        self.runmean = [None for i in range(self.dim)]
        self.lastdim = -1
        self.lastfmin = 1e10


class DimSelectWrapper:
    """
    A generic wrapper around specific dimselect methods that
    performs some common tasks like updating history data,
    burn-in and epsilon-greedy exploration.
    """
    def __init__(self, options, dimselect):
        self.options = options
        self.dimselect = dimselect

    def __call__(self, fun, optimize, niter, min):
        try:
            # For stateful dimselects
            self.dimselect.update(min)
        except:
            pass

        if niter < len(optimize) * options['burnin']:
            # Round-robin - initially
            return niter % len(optimize)

        elif np.random.rand() <= self.options['egreedy']:
            # Random sampling - 1-epsilon frequently
            return np.random.randint(len(optimize))

        else:
            # The proper selection method
            return self.dimselect(fun, optimize, niter, min)


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
        try:
            # For stateful dimselects
            options['dimselect'].reset()
        except:
            pass

    print(globres)
    print(_format_solution(globres, f.optimum))
    return globres


def usage(err=2):
    print('Benchmark ndstep, ndstep_seq')
    print('Usage: test.py [-b BURNIN] [-f {f4,bFID}] [-d DIM] [-e {rr,random,mindiff,maxdiff,diffpd,rdiffpd}] [-g EPSILON] [-i MAXITER] [-s SEED] [-r REPEATS] {ndstep,ndstep_seq}')
    sys.exit(err)


if __name__ == "__main__":
    # Deal with options and such

    options = {
        'f': F4,
        'dim': 20,
        'maxiter': 32000,
        'seed': 43,
        'dimselect': None,
        'egreedy': 0.5,
        'burnin': 4,  # *D iters are spend systematically sampling first
    }
    repeats = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "b:d:e:f:g:hi:r:s:", ["help"])
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
                             diffpd=dimselect_diffpd, rdiffpd=dimselect_rdiffpd,
                             history='history', historyRA='historyRA')
            options['dimselect'] = dimstrats[a]
        elif o == "-f":
            if a == "f4":
                options['f'] = F4
            elif a.startswith('b'):
                options['f'] = BBOBFactory(int(a[1:]))
            else:
                usage()
        elif o == "-g":
            options['egreedy'] = float(a)
        elif o == "-b":
            options['burnin'] = int(a)
        elif o == "-d":
            options['dim'] = int(a)
        elif o == "-i":
            options['maxiter'] = int(a)
        elif o == "-r":
            repeats = int(a)
        elif o == "-s":
            options['seed'] = int(a)
        else:
            assert False, "unhandled option"

    method = args[0]

    if options['dimselect'] == 'history':
        options['dimselect'] = DimSelectHistory(options['dim'])
    elif options['dimselect'] == 'historyRA':
        options['dimselect'] = DimSelectHistoryRA(options['dim'])

    if options['dimselect'] is not None:
        options['dimselect'] = DimSelectWrapper(options, options['dimselect'])

    # Now, actually run the circus!

    globres_list = []
    for i in range(repeats):
        if method == "ndstep":
            globres = run_ndstep('ndstep-log.txt', ndstep_minimize, options)
        elif method == "ndstep_seq":
            globres = run_ndstep('ndstep_seq-log.txt', ndstep_seq_minimize, options)
        else:
            assert False
        globres_list.append(globres)
        options['seed'] += 13

    if repeats > 1:
        globres_conv = filter(lambda gr: gr['fun'] <= 1e-8, globres_list)
        conv_ratio = float(len(globres_conv)) / len(globres_list)
        nits = np.array([gr['nit'] for gr in globres_conv])
        restarts = np.array([gr['restarts'] for gr in globres_conv])
        print('% 3.1f%% converged, conv. average nit=%.1f +-%.1f, restarts=%.1f +-%.1f' %
              (conv_ratio * 100,
               np.mean(nits), np.std(nits),
               np.mean(restarts), np.std(restarts)))
