EPS = 1e-12


def max_flooring(x, threshold=EPS):
    x[x < threshold] = threshold
    return x


def add_flooring(x, threshold=EPS):
    x = x + threshold
    return x
