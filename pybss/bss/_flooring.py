EPS = 1e-12


def max_flooring(x, eps=EPS):
    x[x < eps] = eps
    return x


def add_flooring(x, eps=EPS):
    x = x + eps
    return x
