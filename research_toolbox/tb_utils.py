import functools
import inspect
import pprint
import itertools
import numpy as np


def powers_of_two(starting_power, ending_power, is_int_type=False):
    assert ending_power >= starting_power
    return np.logspace(starting_power,
                       ending_power,
                       num=ending_power - starting_power + 1,
                       base=2,
                       dtype='float64' if is_int_type is False else 'int64')


### auxiliary functions for avoiding boilerplate in in assignments
def set_object_variables(obj, d, abort_if_exists=False,
                         abort_if_notexists=True):
    d_to = vars(obj)
    assert (not abort_if_exists) or all([k not in d_to for k in d.keys()])
    assert (not abort_if_notexists) or all([k in d_to for k in d.keys()])

    for k, v in d.items():
        assert not hasattr(obj, k)
        setattr(obj, k, v)
    return obj


def get_object_variables(obj, varnames, tuple_fmt=False):
    d = vars(obj)
    if tuple_fmt:
        return tuple([d[k] for k in varnames])
    else:
        return subset_dict_via_selection(d, varnames)


### partial application and other functional primitives
def partial_apply(fn, d):
    return functools.partial(fn, **d)


def to_list_fn(f):
    return lambda xs: list(map(f, xs))


def transform(x, fns):
    for f in fns:
        x = f(x)
    return x


def zip_toggle(xs):
    """[[x1, ...], [x2, ...], [x3, ...]] --> [(x1, x2, .., xn) ...];
        [(x1, x2, .., xn) ...] --> [[x1, ...], [x2, ...], [x3, ...]]"""
    assert isinstance(xs, list)
    return list(zip(*xs))


### useful iterators
def iter_product(lst_lst_vals, tuple_fmt=True):
    vs = list(itertools.product(*lst_lst_vals))
    if not tuple_fmt:
        vs = list(map(list, vs))
    return vs


def iter_ortho_all(lst_lst_vals, reference_idxs, ignore_repeats=True):
    assert len(lst_lst_vals) == len(reference_idxs)
    ref_r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(reference_idxs)]

    # put reference first in this case, if ignoring repeats
    rs = [] if not ignore_repeats else [tuple(ref_r)]

    num_lsts = len(lst_lst_vals)
    for i in range(num_lsts):
        num_vals = len(lst_lst_vals[i])
        for j in range(num_vals):
            if ignore_repeats and j == reference_idxs[i]:
                continue

            r = list(ref_r)
            r[i] = lst_lst_vals[i][j]
            rs.append(tuple(r))
    return rs


def iter_ortho_single(lst_lst_vals,
                      reference_idxs,
                      iteration_idx,
                      put_reference_first=True):
    assert len(lst_lst_vals) == len(reference_idxs)
    ref_r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(reference_idxs)]

    rs = [] if not put_reference_first else [tuple(ref_r)]

    num_vals = len(lst_lst_vals[iteration_idx])
    for j in range(num_vals):
        if put_reference_first and j == reference_idxs[iteration_idx]:
            continue

        r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(reference_idxs)]
        r[i] = lst_lst_vals[iteration_idx][j]
        rs.append(tuple(r))
    return rs


def get_argument_names(fn):
    return inspect.getargspec(fn).args


### dictionary manipulation
def create_dict(ks, vs):
    assert len(ks) == len(vs)
    return dict(list(zip(ks, vs)))


def create_dataframe(ds, abort_if_different_keys=True):
    import pandas
    ks = key_union(ds)
    assert (not abort_if_different_keys) or len(key_intersection(ds)) == len(ks)

    df_d = {k: [] for k in ks}
    for d in ds:
        for k in ks:
            if k not in d:
                df_d[k].append(None)
            else:
                df_d[k].append(d[k])

    df = pandas.DataFrame(df_d)
    return df


def copy_update_dict(d, d_other):
    proc_d = dict(d)
    proc_d.update(d_other)
    return proc_d


def merge_dicts(ds):
    out_d = {}
    for d in ds:
        for (k, v) in d.items():
            assert k not in out_d
            out_d[k] = v
    return out_d


def groupby(xs, fn):
    assert isinstance(xs, list)

    d = {}
    for x in xs:
        fx = fn(x)
        if fx not in d:
            d[fx] = []

        d[fx].append(x)
    return d


def flatten(d):
    assert isinstance(d, dict)

    xs = []
    for (_, k_xs) in d.items():
        assert isinstance(k_xs, list)
        xs.extend(k_xs)
    return xs


def recursive_groupby(p, fn):
    assert isinstance(p, (dict, list))

    if isinstance(p, list):
        return groupby(p, fn)
    else:
        return {k: recursive_groupby(k_p, fn) for (k, k_p) in p.items()}


def recursive_flatten(p):
    assert isinstance(p, (dict, list))

    if isinstance(p, list):
        return list(p)
    else:
        xs = []
        for (_, k_p) in p.items():
            xs.extend(recursive_flatten(k_p))
        return xs


def recursive_map(p, fn):
    assert isinstance(p, (dict, list))

    if isinstance(p, list):
        return list(map(fn, p))
    else:
        return {k: recursive_map(k_p, fn) for (k, k_p) in p.items()}


def recursive_index(d, ks):
    for k in ks:
        d = d[k]
    return d


def filter_dict(d, fn):
    return {k: v for (k, v) in d.items() if fn(k, v)}


def map_dict(d, fn):
    return {k: fn(k, v) for (k, v) in d.items()}


def invert_injective_dict(a_to_b):
    b_to_a = dict([(b, a) for (a, b) in a_to_b.items()])
    return b_to_a


def invert_noninjective_dict(a_to_b):
    d = {}
    for (x, y) in a_to_b.items():
        if y not in d:
            d[y] = []
        d[y].append(x)
    return d


def structure(ds, ks):
    get_fn = lambda k: lambda x: x[k]

    for k in ks:
        ds = recursive_groupby(ds, get_fn(k))
    return ds


def structure_with_fns(ds, fns):
    for fn in fns:
        ds = recursive_groupby(ds, fn)
    return ds


def get_subset_indexing_fn(ks, tuple_fmt=True):

    def fn(d):
        assert isinstance(d, dict)
        out = [d[k] for k in ks]
        if tuple_fmt:
            out = tuple(out)
        return out

    return fn


def flatten_nested_list(xs):
    assert isinstance(xs, list)

    xs_res = []
    for x_lst in xs:
        assert isinstance(x_lst, list)
        xs_res.extend(x_lst)
    return xs_res


def key_union(ds):
    ks = []
    for d in ds:
        ks.extend(list(d.keys()))
    return list(set(ks))


def key_intersection(ds):
    assert len(ds) > 0

    ks = set(ds[0].keys())
    for d in ds[1:]:
        ks.intersection_update(list(d.keys()))
    return list(ks)


# NOTE: right now, done for dictionaries with hashable values.
def key_to_values(ds):
    out_d = {}
    for d in ds:
        for (k, v) in d.items():
            if k not in out_d:
                out_d[k] = set()
            out_d[k].add(v)
    return out_d


# NOTE: for now, no checking regarding the existence of the key.
def subset_dict_via_selection(d, ks):
    return {k: d[k] for k in ks}


def subset_dict_via_deletion(d, ks):
    out_d = dict(d)
    for k in ks:
        out_d.pop(k)
    return out_d


def set_dict_values(d_to,
                    d_from,
                    abort_if_exists=False,
                    abort_if_notexists=True):
    assert (not abort_if_exists) or all([k not in d_to for k in d_from.keys()])
    assert (not abort_if_notexists) or all([k in d_to for k in d_from.keys()])

    d_to.update(d_from)


def sort_dict_items(d, by_key=True, decreasing=False):
    key_fn = (lambda x: x[0]) if by_key else (lambda x: x[1])
    return sorted(list(d.items()), key=key_fn, reverse=decreasing)


def print_dict(d, width=1):
    pprint.pprint(d, width=width)


def pprint_dict_keys(d, ks):
    pprint.pprint({k: d[k] for k in ks})


def collapse_nested_dict(d, sep='.'):
    assert all([type(k) == str for k in d.keys()]) and (all(
        [all([type(kk) == str for kk in d[k].keys()]) for k in d.keys()]))

    ps = []
    for k in d.keys():
        for (kk, v) in d[k].items():
            p = (k + sep + kk, v)
            ps.append(p)
    return dict(ps)


def uncollapse_nested_dict(d, sep='.'):
    assert all([type(k) == str for k in d.keys()]) and (all(
        [len(k.split()) == 2 for k in d.keys()]))

    out_d = []
    for (k, v) in d.items():
        (k1, k2) = k.split(sep)
        if k1 not in out_d:
            d[k1] = {}
        d[k1][k2] = v
    return out_d
