### simple io
import json
import pickle
import research_toolbox.tb_utils as tb_ut


def read_textfile(filepath, strip=True):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if strip:
            lines = [line.strip() for line in lines]
        return lines


def write_textfile(filepath, lines, append=False, with_newline=True):
    mode = 'a' if append else 'w'

    with open(filepath, mode) as f:
        for line in lines:
            f.write(line)
            if with_newline:
                f.write("\n")


def read_dictfile(filepath, sep='\t'):
    """This function is conceived for dicts with string keys."""
    lines = read_textfile(filepath)
    d = dict([line.split(sep, 1) for line in lines])
    return d


def write_dictfile(filepath, d, sep="\t", ks=None):
    """This function is conceived for dicts with string keys. A set of keys
    can be passed to specify an ordering to write the keys to disk.
    """
    if ks == None:
        ks = list(d.keys())
    assert all([type(k) == str and sep not in k for k in ks])
    lines = [sep.join([k, str(d[k])]) for k in ks]
    write_textfile(filepath, lines)


def read_jsonfile(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
        return d


def write_jsonfile(d, filepath, sort_keys=False, compactify=False):
    with open(filepath, 'w') as f:
        indent = None if compactify else 4
        json.dump(d, f, indent=indent, sort_keys=sort_keys)


def read_jsonfile_with_overlays(filepath):
    d = read_jsonfile(filepath)
    d_out = {}
    if "_overlays_" in d:
        for fp in d["_overlays_"]:
            d_other = read_jsonfile_with_overlays(fp)
            d_out.update(d_other)
    d_out.update(d)
    return d_out


def json2key(d):
    return json.dumps(d, sort_keys=True)


def json2str(d):
    return json.dumps(d)


def str2json(s):
    return json.loads(s)


def read_jsonlogfile(filepath):
    with open(filepath, 'r') as f:
        return [str2json(line) for line in f]


def write_jsonlogfile(filepath, ds, append=False):
    write_textfile(
        filepath, list(map(json2str, ds)), append=append, with_newline=True)


def read_picklefile(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def write_picklefile(x, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(x, f)


# NOTE: this function can use some existing functionality from python.
def read_csvfile(filepath, sep=',', has_header=True):
    raise NotImplementedError


# TODO: there is also probably some functionality for this.
def write_csvfile(ds,
                  filepath,
                  sep=',',
                  write_header=True,
                  abort_if_different_keys=True,
                  sort_keys=False):
    ks = tb_ut.key_union(ds)
    if sort_keys:
        ks.sort()

    assert (not abort_if_different_keys) or len(
        tb_ut.key_intersection(ds)) == len(ks)

    lines = []
    if write_header:
        lines.append(sep.join(ks))

    for d in ds:
        lines.append(sep.join([str(d[k]) if k in d else '' for k in ks]))

    write_textfile(filepath, lines)
