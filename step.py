import math
from operator import itemgetter
from scipy import optimize


def interval_difficulty(points, values, fmin, epsilon):
    """
    Compute difficulty of a single interval between two points.
    """
    # Recompute the second point coordinates with regards to the left (first)
    # point.
    x = points[1] - points[0]
    y = values[1] - values[0]
    f = fmin - values[0] - epsilon

    # Curvature of parabole crossing [0,0], [x,y] and touching [?, f]
    a = (y - 2*f + 2*math.sqrt(f * (f - y))) / (x**2)
    return a


def recompute_difficulty(points, values, fmin, epsilon):
    difficulty = []
    for i in range(len(points) - 1):
        difficulty.append(interval_difficulty(points[i:i+2], values[i:i+2], fmin, epsilon))
    return difficulty


def step(fun, bounds, args=(), maxiter=100, callback=None, epsilon=1e-8, disp=False, tolx=1e-11, **options):
    """
    TODO
    """

    points = [bounds[0], (bounds[0] + bounds[1]) / 2.0, bounds[1]]
    values = map(lambda p: fun(p), points)
    imin, fmin = min(enumerate(values), key=itemgetter(1))
    xmin = points[imin]
    difficulty = recompute_difficulty(points, values, fmin, epsilon)

    niter = 0
    while niter < maxiter:
        # Select the easiest interval which is wide enough
        idiff = filter(lambda (i, diff): points[i+1] - points[i] >= tolx,
                       enumerate(difficulty))
        if len(idiff) == 0:
            break  # We cannot split the interval more
        i, diff = min(idiff, key=itemgetter(1))

        if disp:
            print(points)
            print(values)
            print(difficulty)
            print('Easiest interval %f: [%f, %f]' % (diff, points[i], points[i+1]))

        # Split it into two
        newpoint = (points[i] + points[i+1]) / 2.0
        newvalue = fun(newpoint)
        points.insert(i+1, newpoint)
        values.insert(i+1, newvalue)
        difficulty[i] = None
        difficulty.insert(i+1, None)
        niter += 1

        if newvalue < fmin:
            # New fmin, recompute difficulties of all intervals
            fmin = newvalue
            xmin = points[i+1]
            difficulty = recompute_difficulty(points, values, fmin, epsilon)
        else:
            # No fmin change, compute difficulties only of the two
            # new intervals
            difficulty[i] = interval_difficulty(points[i:i+2], values[i:i+2], fmin, epsilon)
            difficulty[i+1] = interval_difficulty(points[i+1:i+3], values[i+1:i+3], fmin, epsilon)

        if callback is not None:
            if callback(xmin):
                break

    return optimize.OptimizeResult(fun=fmin, x=xmin, nit=niter,
                                   success=(niter > 1))
