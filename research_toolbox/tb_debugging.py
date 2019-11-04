### simple tools useful in implementing simple correctness checks.
import itertools


def test_overfit(ds, overfit_val, overfit_key):
    ov_ds = []
    err_ds = []
    for d in ds:
        if d[overfit_key] == overfit_val:
            ov_ds.append(d)
        else:
            err_ds.append(d)

    return (ov_ds, err_ds)


def test_with_fn(ds, fn):
    good_ds = []
    bad_ds = []
    for d in ds:
        if fn(d):
            good_ds.append(d)
        else:
            bad_ds.append(d)

    return (good_ds, bad_ds)


# assumes that eq_fn is transitive.
def all_equivalent_with_fn(ds, eq_fn):
    for d1, d2 in zip(ds[:-1], ds[1:]):
        if not eq_fn(d1, d2):
            return False
    return True


def assert_length_consistency(xs_lst):
    assert len(set(map(len, xs_lst))) == 1

    for i in range(len(xs_lst)):
        assert set([len(xs[i]) for xs in xs_lst]) == 1


def is_at_most_one_true(xs):
    assert all(type(x) == bool for x in xs)
    return sum(xs) <= 1
