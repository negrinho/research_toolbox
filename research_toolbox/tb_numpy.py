import numpy as np


def vectorial_indices_where_true(x):
    assert len(x.shape) == 1
    return x.nonzero()[0]


def tensorial_indices_where_true(x):
    return x.nonzero()

def logsumexp(x):
    m = np.max(x, axis=-1)
    z = np.log(np.sum(np.exp(x - m[:, None]), axis=-1))
    return m + z


def shapes_of_all_tensors(scope):
    """Can be called by passing vars()."""
    out = {}
    for name, v in scope.items():
        if isinstance(v, np.ndarray):
            out[name] = v.shape
    return out

# TODO: some basic vector norms.