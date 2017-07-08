
### auxiliary functions for avoiding boilerplate in in assignments
def set_obj_vars(obj, d, 
        abort_if_exists=False, abort_if_notexists=True):
    d_to = vars(obj)
    assert (not abort_if_exists) or all(
        [k not in d_to for k in d.iterkeys()])
    assert (not abort_if_notexists) or all(
        [k in d_to for k in d.iterkeys()])    

    for k, v in d.iteritems():
        assert not hasattr(obj, k) 
        setattr(obj, k, v)
     
def retrieve_obj_vars(obj, var_names, tuple_fmt=False):
    return retrieve_values(vars(obj), var_names, tuple_fmt)

### partial application and other functional primitives
import functools

def partial_apply(fn, d):
    return functools.partial(fn, **d)

def to_list_fn(f):
    return lambda xs: map(f, xs)

def transform(x, fns):
    for f in fns:
        x = f( x )
    return y

### dictionary manipulation
import pprint
import pandas

def create_dict(ks, vs):
    assert len(ks) == len(vs)
    return dict(zip(ks, vs))

def create_dataframe(ds, abort_if_different_keys=True):
    ks = key_union(ds)
    assert (not abort_if_different_keys) or len( key_intersection(ds) ) == len( ks ) 

    df_d = {k : [] for k in ks }
    for d in ds:
        for k in ks:
            if k not in d:
                df_d[k].append( None )
            else:
                df_d[k].append( d[k] )
    
    df = pandas.DataFrame(df_d)
    return df

def copy_update_dict(d, d_other):
    proc_d = dict(d)
    proc_d.update(d_other)

    return proc_d

def merge_dicts(ds):
    out_d = {}
    for d in ds:
        for (k, v) in d.iteritems():
            assert k not in out_d
            out_d[k] = v
    return out_d

def groupby(xs, fn):
    assert isinstance(xs, list)  
    
    d = {}
    for x in xs:
        fx = fn( x )
        if fx not in d:
            d[fx] = []

        d[fx].append(x)
    return d

def flatten(d):
    assert isinstance(d, dict)

    xs = []
    for (_, k_xs) in d.iteritems():
        assert isinstance(k_xs, list)
        xs.extend(k_xs)
    return xs

def recursive_groupby(p, fn):
    assert isinstance(p, dict) or isinstance(p, list)

    if isinstance(p, list):
        return groupby(p, fn)
    else:
        return {k : recursive_groupby(k_p, fn) for (k, k_p) in p.iteritems()}

def recursive_flatten(p):
    assert isinstance(p, dict) or isinstance(p, list)

    if isinstance(p, list):
        return list(p)
    else:
        xs = []
        for (_, k_p) in p.iteritems():
            xs.extend( recursive_flatten(k_p) )
        return xs

def recursive_map(p, fn):
    assert isinstance(p, dict) or isinstance(p, list)

    if isinstance(p, list):
        return map(fn, p)
    else:
        return {k : recursive_map(k_p, fn) for (k, k_p) in p.iteritems()}

def recursive_index(d, ks):
    for k in ks:
        d = d[k]
    return d

def filter_dict(d, fn):
    return { k : v for (k, v) in d.iteritems() if fn(k, v) }

def map_dict(d, fn):
    return { k : fn(k, v) for (k, v) in d.iteritems() }

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

def zip_toggle(xs):
    """[[x1, ...], [x2, ...], [x3, ...]] --> [(x1, x2, .., xn) ... ];
        [(x1, x2, .., xn) ... ] --> [[x1, ...], [x2, ...], [x3, ...]]"""
    assert isinstance(xs, list)
    return zip(*xs)

def key_union(ds):
    ks = []
    for d in ds:
        ks.extend(d.keys())
    return list(set(ks))

def key_intersection(ds):
    assert len(ds) > 0

    ks = set(ds[0].keys())
    for d in ds[1:]:
        ks.intersection_update(d.keys())
    return list(ks)

# NOTE: right now, this is done for dictionaries with hashable values.
def key_to_values(ds):
    out_d = {}
    for d in ds:
        for (k, v) in d.iteritems():
            if k not in out_d:
                out_d[k] = set()
            out_d[k].add(v)
    return out_d

def retrieve_values(d, ks, tuple_fmt=False):
    out_d = {}
    for k in ks:
        out_d[k] = d[k]
    
    if tuple_fmt:
        out_d = tuple([out_d[k] for k in ks])
    return out_d

def set_values(d_to, d_from, 
        abort_if_exists=False, abort_if_notexists=True):
    assert (not abort_if_exists) or all(
        [k not in d_to for k in d_from.iterkeys()])
    assert (not abort_if_notexists) or all(
        [k in d_to for k in d_from.iterkeys()])
    
    d_to.update(d_from)

def sort_items(d, by_key=True, decreasing=False):
    key_fn = (lambda x: x[0]) if by_key else (lambda x: x[1])
    return sorted(d.items(), key=key_fn, reverse=decreasing)

def print_dict(d, width=1):
    pprint.pprint(d, width=width)

def collapse_nested_dict(d, sep='.'):
    assert all([type(k) == str for k in d.iterkeys()]) and (
        all([all([type(kk) == str for kk in d[k].iterkeys()]) for k in d.iterkeys()]))
    
    ps = []
    for k in d.iterkeys():
        for (kk, v) in d[k].iteritems():
            p = (k + sep + kk, v)
            ps.append(p)
    return dict(ps)

def uncollapse_nested_dict(d, sep='.'):
    assert all([type(k) == str for k in d.iterkeys()]) and (
        all([len(k.split()) == 2 for k in d.iterkeys()]))

    out_d = []
    for (k, v) in d.iteritems():
        (k1, k2) = k.split()
        if k1 not in out_d:
            d[k1] = {}
        d[k1][k2] = v

    return out_d

### simple io
import json
import sys
import pickle

def read_textfile(fpath, strip=True):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        if strip:
            lines = [line.strip() for line in lines]
        return lines

def write_textfile(fpath, lines, append=False, with_newline=True):
    mode = 'a' if append else 'w'

    with open(fpath, mode) as f:
        for line in lines:
            f.write(line)
            if with_newline:
                f.write("\n")

def read_dictfile(fpath, sep='\t'):
    """This function is conceived for dicts with string keys."""
    lines = read_textfile(fpath)
    d = dict([lines.split(sep, 1) for line in lines])
    return d

def write_dictfile(fpath, d, sep="\t", ks=None):
    """This function is conceived for dicts with string keys. A set of keys 
    can be passed to specify an ordering to write the keys to disk.
    """
    if ks == None:
        ks = d.keys()
    assert all([type(k) == str and sep not in k for k in ks])
    lines = [sep.join([k, str(d[k])]) for k in ks]
    write_textfile(fpath, lines)

def read_jsonfile(fpath):
    with open(fpath, 'r') as f:
        d = json.load(f)
        return d

def write_jsonfile(d, fpath, sort_keys=False):
    with open(fpath, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=sort_keys)

def read_picklefile(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)

def write_picklefile(x, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(x, f)

# NOTE: this function can use some existing functionality from python to 
# make my life easier. I don't want fields manually. 
# perhaps easier when supported in some cases.
def read_csvfile(fpath, sep=',', has_header=True):
    raise NotImplementedError

# TODO: there is also probably some functionality for this.
def write_csvfile(ds, fpath, sep=',',
        write_header=True, abort_if_different_keys=True, sort_keys=False):

    ks = key_union(ds)
    if sort_keys:
        ks.sort()

    assert (not abort_if_different_keys) or len( key_intersection(ds) ) == len(ks)

    lines = []
    if write_header:
        lines.append( sep.join(ks) )
    
    for d in ds:
        lines.append( sep.join([str(d[k]) if k in d else '' for k in ks]) )

    write_textfile(fpath, lines)

# check if files are flushed automatically upon termination, or there is 
# something else that needs to be done. or maybe have a flush flag or something.
def capture_output(f, capture_stdout=True, capture_stderr=True):
    """Takes a file as argument. The file is managed externally."""
    if capture_stdout:
        sys.stdout = f
    if capture_stderr:
        sys.stderr = f

### directory management.
import os
import shutil

def path_prefix(path):
    return os.path.split(path)[0]

def path_last_element(path):
    return os.path.split(path)[1]

def path_relative_to_absolute(path):
    return os.path.abspath(path)

def path_exists(path):
    return os.path.exists(path)

def file_exists(path):
    return os.path.isfile(path)

def folder_exists(path):
    return os.path.isdir(path)

def create_file(filepath,
        abort_if_exists=True, create_parent_folders=False):
    assert create_parent_folders or folder_exists(path_prefix(filepath))
    assert not (abort_if_exists and file_exists(filepath))

    if create_parent_folders:
        create_folder(path_prefix(filepath),
            abort_if_exists=False, create_parent_folders=True)

    with open(filepath, 'w'):
        pass

def create_folder(folderpath, 
        abort_if_exists=True, create_parent_folders=False):

    assert not file_exists(folderpath)
    assert create_parent_folders or folder_exists(path_prefix(folderpath))
    assert not (abort_if_exists and folder_exists(folderpath))

    if not folder_exists(folderpath):
        os.makedirs(folderpath)

def copy_file(src_filepath, dst_filepath, 
        abort_if_dst_exists=True, create_parent_folders=False):

    # print src_filepath
    assert file_exists(src_filepath)
    assert src_filepath != dst_filepath
    assert not (abort_if_dst_exists and file_exists(dst_filepath))  

    src_filename = path_last_element(src_filepath)
    dst_folderpath = path_prefix(dst_filepath)
    dst_filename = path_last_element(dst_filepath)
    
    assert create_parent_folders or folder_exists(dst_folderpath)
    if not folder_exists(dst_folderpath):
        create_folder(dst_folderpath, create_parent_folders=True)

    shutil.copyfile(src_filepath, dst_filepath)

def copy_folder(src_folderpath, dst_folderpath, 
        ignore_hidden_files=False, ignore_hidden_folders=False, ignore_file_exts=None,
        abort_if_dst_exists=True, create_parent_folders=False):

    assert folder_exists(src_folderpath)
    assert src_folderpath != dst_folderpath
    assert not (abort_if_dst_exists and folder_exists(dst_folderpath))  

    if (not abort_if_dst_exists) and folder_exists(dst_folderpath):
        delete_folder(dst_folderpath, abort_if_nonempty=False)
    
    pref_dst_fo = path_prefix(dst_folderpath)
    assert create_parent_folders or folder_exists(pref_dst_fo)
    create_folder(dst_folderpath, create_parent_folders=create_parent_folders)
    
    # create all folders in the destination.
    args = retrieve_values(locals(), 
        ['ignore_hidden_folders', 'ignore_hidden_files'])  
    fos = list_folders(src_folderpath, use_relative_paths=True, recursive=True, **args)

    for fo in fos:
        fo_path = join_paths([dst_folderpath, fo])
        create_folder(fo_path, create_parent_folders=True)
    # print fos

    # copy all files to the destination. 
    args = retrieve_values(locals(), 
        ['ignore_hidden_folders', 'ignore_hidden_files', 'ignore_file_exts'])  
    fis = list_files(src_folderpath, use_relative_paths=True, recursive=True, **args)
    # print fis

    for fi in fis:
        src_fip = join_paths([src_folderpath, fi])
        dst_fip = join_paths([dst_folderpath, fi])
        copy_file(src_fip, dst_fip)

def delete_file(filepath, abort_if_notexists=True):
    assert file_exists(filepath) or (not abort_if_notexists)
    if file_exists(filepath):
        os.remove(filepath)

def delete_folder(folderpath, abort_if_nonempty=True, abort_if_notexists=True):

    assert folder_exists(folderpath) or (not abort_if_notexists)

    if folder_exists(folderpath):
        assert len(os.listdir(folderpath)) == 0 or (not abort_if_nonempty)
        shutil.rmtree(folderpath)
    else:
        assert not abort_if_notexists

def list_paths(folderpath, 
        ignore_files=False, ignore_dirs=False, 
        ignore_hidden_folders=True, ignore_hidden_files=True, ignore_file_exts=None, 
        recursive=False, use_relative_paths=False):

    assert folder_exists(folderpath)
    
    path_list = []
    # enumerating all desired paths in a directory.
    for root, dirs, files in os.walk(folderpath):
        if ignore_hidden_folders:
            dirs[:] = [d for d in dirs if not d[0] == '.']
        if ignore_hidden_files:
            files = [f for f in files if not f[0] == '.']
        if ignore_file_exts != None:
            # print files
            files = [f for f in files if not any([
                f.endswith(ext) for ext in ignore_file_exts])]

        # get only the path relative to this path.
        if not use_relative_paths: 
            pref_root = root 
        else:
            pref_root = os.path.relpath(root, folderpath)

        if not ignore_files:
            path_list.extend([join_paths([pref_root, f]) for f in files])
        if not ignore_dirs:
            path_list.extend([join_paths([pref_root, d]) for d in dirs])

        if not recursive:
            break
    return path_list

def list_files(folderpath, 
        ignore_hidden_folders=True, ignore_hidden_files=True, ignore_file_exts=None, 
        recursive=False, use_relative_paths=False):

    args = retrieve_values(locals(), ['recursive', 'ignore_hidden_folders', 
        'ignore_hidden_files', 'ignore_file_exts', 'use_relative_paths'])

    return list_paths(folderpath, ignore_dirs=True, **args)

def list_folders(folderpath, 
        ignore_hidden_files=True, ignore_hidden_folders=True, 
        recursive=False, use_relative_paths=False):

    args = retrieve_values(locals(), ['recursive', 'ignore_hidden_folders', 
        'ignore_hidden_files', 'use_relative_paths'])

    return list_paths(folderpath, ignore_files=True, **args)

def join_paths(paths):
    return os.path.join(*paths)

def pairs_to_filename(ks, vs, kv_sep='', pair_sep='_', prefix='', suffix='.txt'):
    pairs = [kv_sep.join([k, v]) for (k, v) in zip(ks, vs)]
    s = prefix + pair_sep.join(pairs) + suffix
    return s

### NOTE: can be done in two parts of the model or not.
def compress_path(path, out_filepath):
    raise NotImplementedError
    # shutil.make_archive(path, 'zip', path)
    # combine with moving functionality, I think.

def decompress_path(path, out_filename):
    raise NotImplementedError
    # pass
    # shutil.ma
# TO delete.
# import shutil
# shutil.make_archive(output_filename, 'zip', dir_name)

### simple tools for randomization
import random

def shuffle(xs):
    idxs = range( len(xs) )
    random.shuffle(idxs)

    return [xs[i] for i in idxs]

def random_permutation(n):
    idxs = range( n )
    random.shuffle(idxs)

    return idxs

def argsort(xs, fns, increasing=True):
    """The functions in fns are used to compute a key which are then used to 
    construct a tuple which is then used to sort. The earlier keys are more 
    important than the later ones.
    """

    def key_fn(x):
        return tuple( [f(x) for f in fns] )
    
    idxs, _ = zip_toggle( 
        sorted( enumerate( xs ), 
            key=lambda x: key_fn( x[1] ), 
            reverse=not increasing ) )

    return idxs

def sort(xs, fns, increasing=True):
    idxs = argsort(xs, fns, increasing)
    return apply_permutation(xs, idxs)

def apply_permutation(xs, idxs):
    assert len( set(idxs).intersection( range( len(xs) ) ) ) == len(xs)

    return [xs[i] for i in idxs]

def apply_inverse_permutation(xs, idxs):
    assert len( set(idxs).intersection( range( len(xs) ) ) ) == len(xs)
    
    out_xs = [None] * len(xs)
    for i_from, i_to in enumerate(idxs):
        out_xs[i_to] = xs[i_from]
    
    return out_xs

def shuffle_tied(xs_lst):
    assert len(xs_lst) > 0 and len(map(len, xs_lst)) == 1

    n = len( xs_lst[0] )
    idxs = random_permutation(n)
    ys_lst = [apply_permutation(xs, idxs) for xs in xs_lst]
    return ys_lst

### date and time
import datetime

def convert_between_time_units(x, src_units='s', dst_units='h'):
    units = ['s', 'm', 'h', 'd', 'w']
    assert src_units in units and dst_units in units
    d = {}
    d['s'] = 1.0
    d['m'] = 60.0 * d['s']
    d['h'] = 60.0 * d['m']
    d['d'] = 24.0 * d['h']
    d['w'] = 7.0 * d['d']
    return (x * d[src_units]) / d[dst_units]

# TODO: do not just return a string representation, return the numbers.
def now(omit_date=False, omit_time=False, time_before_date=False):
    assert (not omit_time) or (not omit_date)

    d = datetime.datetime.now()

    date_s = ''
    if not omit_date:
        date_s = "%d-%.2d-%.2d" % (d.year, d.month, d.day)
    time_s = ''
    if not omit_time:
        time_s = "%.2d:%.2d:%.2d" % (d.hour, d.minute, d.second)
    
    vs = []
    if not omit_date:
        vs.append(date_s)
    if not omit_time:
        vs.append(time_s)

    # creating the string
    if len(vs) == 2 and time_before_date:
        vs = vs[::-1]
    s = '|'.join(vs)

    return s

def now_dict():
    x = datetime.datetime.now()

    return  create_dict(
        ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
        [x.year, x.month, x.day, x.hour, x.minute, x.second, x.microsecond])

### useful iterators
import itertools

def iter_product(lst_lst_vals, tuple_fmt=True):
    vs = list(itertools.product(*lst_lst_vals))
    if not tuple_fmt:
        vs = map(list, vs)

    return vs

def iter_ortho_all(lst_lst_vals, ref_idxs, ignore_repeats=True):
    assert len(lst_lst_vals) == len(ref_idxs)
    ref_r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(ref_idxs)]

    # put reference first in this case, if ignoring repeats
    rs = [] if not ignore_repeats else [tuple(ref_r)]

    num_lsts = len(lst_lst_vals)
    for i in xrange( num_lsts ):
        num_vals = len(lst_lst_vals[i])
        for j in xrange( num_vals ):
            if ignore_repeats and j == ref_idxs[i]:
                continue

            r = list(ref_r)
            r[i] = lst_lst_vals[i][j]
            rs.append(tuple(r))
    return rs

def iter_ortho_single(lst_lst_vals, ref_idxs, idx_it, ref_first=True):
    assert len(lst_lst_vals) == len(ref_idxs)
    ref_r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(ref_idxs)]
    
    rs = [] if not ref_first else [tuple(ref_r)]

    num_vals = len(lst_lst_vals[idx_it])
    for j in xrange( num_vals ):
        if ref_first and j == ref_idxs[idx_it]:
            continue

        r = [lst_lst_vals[pos][idx] for (pos, idx) in enumerate(ref_idxs)]
        r[i] = lst_lst_vals[idx_it][j]
        rs.append(tuple(r))
    return rs

### logging
import time
import os

### TODO: add calendar. or clock. or both. check formatting for both.

class TimeTracker:
    def __init__(self):
        self.init_time = time.time()
        self.last_registered = self.init_time
    
    def time_since_start(self, units='s'):
        now = time.time()
        elapsed = now - self.init_time
        
        return convert_between_time_units(elapsed, dst_units=units)
        
    def time_since_last(self, units='s'):
        now = time.time()
        elapsed = now - self.last_registered
        self.last_registered = now

        return convert_between_time_units(elapsed, dst_units=units)            

class MemoryTracker:
    def __init__(self):
        self.last_registered = 0.0
        self.max_registered = 0.0

    def memory_total(self, units='mb'):
        mem_now = mbs_process(os.getpid())
        if self.max_registered < mem_now:
            self.max_registered = mem_now

        return convert_between_byte_units(mem_now, dst_units=units)
        
    def memory_since_last(self, units='mb'):
        mem_now = self.memory_total()        
        
        mem_dif = mem_now - self.last_registered
        self.last_registered = mem_now

        return convert_between_byte_units(mem_dif, dst_units=units)

    def memory_max(self, units='mb'):
        return convert_between_byte_units(self.max_registered, dst_units=units)

def print_time(timer, pref_str='', units='s'):
    print('%s%0.2f %s since start.' % (pref_str, 
        timer.time_since_start(units=units), units) )
    print("%s%0.2f %s seconds since last call." % (pref_str, 
        timer.time_since_last(units=units), units) )

def print_memory(memer, pref_str='', units='mb'):
    print('%s%0.2f %s total.' % (pref_str, 
        memer.memory_total(units=units), units.upper()) )
    print("%s%0.2f %s since last call." % (pref_str, 
        memer.memory_since_last(units=units), units.upper()) )
    
def print_memorytime(memer, timer, pref_str='', mem_units='mb', time_units='s'):
    print_memory(memer, pref_str, units=mem_units)
    print_time(timer, pref_str, units=time_units)    

def print_oneliner_memorytime(memer, timer, pref_str='', 
        mem_units='mb', time_units='s'):

    print('%s (%0.2f %s last; %0.2f %s total; %0.2f %s last; %0.2f %s total)'
                 % (pref_str,
                    timer.time_since_last(units=time_units),
                    time_units, 
                    timer.time_since_start(units=time_units),
                    time_units,                     
                    memer.memory_since_last(units=mem_units),
                    mem_units.upper(), 
                    memer.memory_total(units=mem_units),
                    mem_units.upper()) )

class Logger:
    def __init__(self, fpath, 
        append_to_file=False, capture_all_output=False):
        
        if append_to_file:
            self.f = open(fpath, 'a')
        else:
            self.f = open(fpath, 'w')

        if capture_all_output:
            capture_output(self.f)
    
    def log(self, s, desc=None, preappend_datetime=False):
        
        if preappend_datetime:
            self.f.write( now() + '\n' )
        
        if desc is not None:
            self.f.write( desc + '\n' )

        self.f.write( s + '\n' )

# for keeping track of configurations and results for programs
class ArgsDict:
    def __init__(self, fpath=None):
        self.d = {}
        if fpath != None:
            self._read(fpath)
    
    def set_arg(self, key, val, abort_if_exists=True):
        assert (not self.abort_if_exists) or key not in self.d 
        assert (type(key) == str and len(key) > 0)
        self.d[key] = val
    
    def write(self, fpath):
        write_jsonfile(self.d, fpath)
    
    def _read(self, fpath):
        return load_jsonfile(fpath)

    def get_dict(self):
        return dict(self.d)

class ConfigDict(ArgsDict):
    pass

class ResultsDict(ArgsDict):
    pass

class SummaryDict:
    def __init__(self, fpath=None, abort_if_different_lengths=False):
        self.abort_if_different_lengths = abort_if_different_lengths
        if fpath != None:
            self.d = self._read(fpath)
        else:
            self.d = {}
        self._check_consistency()
    
    def append(self, d):
        for k, v in d.iteritems():
            assert type(k) == str and len(k) > 0
            if k not in self.d:
                self.d[k] = []
            self.d[k].append(v)
        
        self._check_consistency()
    
    ### NOTE: I don't think that read makes sense anymore because you 
    # can keep think as a json file or something else.
    def write(self, fpath):
        write_jsonfile(self.d, fpath)
    
    def _read(self, fpath):
        return load_jsonfile(fpath)
    
    def _check_consistency(self):
        assert (not self.abort_if_different_lengths) or (len(
            set([len(v) for v in self.d.itervalues()])) <= 1)

    def get_dict(self):
        return dict(self.d)

### command line arguments
import argparse

class CommandLineArgs:
    def __init__(self, args_prefix=''):
        self.parser = argparse.ArgumentParser()
        self.args_prefix = args_prefix
    
    def add(self, name, type, default=None, optional=False, help=None,
            valid_choices=None, list_valued=False):
        valid_types = {'int' : int, 'str' : str, 'float' : float}
        assert type in valid_types

        nargs = None if not list_valued else '*'
        type = valid_types[type]

        self.parser.add_argument('--' + self.args_prefix + name, 
            required=not optional, default=default, nargs=nargs, 
            type=type, choices=valid_choices, help=help)

    def parse(self):
        return self.parser.parse_args()
    
    def get_parser(self):
        return self.parser

### running locally and on a server
import psutil
import subprocess

def cpus_total():
    return psutil.cpu_count()

def cpus_free():
    frac_free = (1.0 - 0.01 * psutil.cpu_percent())
    return int( np.round( frac_free * psutil.cpu_count() ) )

def convert_between_byte_units(x, src_units='b', dst_units='mb'):
     units = ['b', 'kb', 'mb', 'gb', 'tb']
     assert (src_units in units) and (dst_units in units)
     return x / float(
         2 ** (10 * (units.index(dst_units) - units.index(src_units))))

def memory_total(units='mb'):
    return convert_between_byte_units(psutil.virtual_memory().total, dst_units=units)
    
def memory_free(units='mb'):
    return convert_between_byte_units(psutil.virtual_memory().available, dst_units=units)

def gpus_total():
    try:
        n = len(subprocess.check_output(['nvidia-smi', '-L']).strip().split('\n'))
    except OSError:
        n = 0
    return n

def gpus_free():
    return len(gpus_free_ids())

def gpus_free_ids():
    try:
        out = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used', 
            '--format=csv,noheader'])

        ids = []

        gpu_ss = out.strip().split('\n')
        for i, s in enumerate( gpu_ss ):
            
            p_s, m_s = s.split(', ')
            p = float( p_s.split()[0] )
            m = float( m_s.split()[0] )

            # NOTE: this is arbitrary for now. maybe a different threshold for 
            # utilization makes sense. maybe it is fine as long as it is not 
            # fully utilized. memory is in MBs.
            if p < 0.1 and m < 100.0:
                ids.append(i)

        return ids

    except OSError:
        ids = []
    
    return ids

def gpus_set_visible(ids):
    n = gpus_total()
    assert all([i < n and i >= 0 for i in ids])
    subprocess.call(['export', 'CUDA_VISIBLE_DEVICES=%s' % ",".join(map(str, ids))])

### download and uploading
import subprocess
import os

def remote_path(username, servername, path):
    return "%s@%s:%s" % (username, servername, path)

# NOTE: this may complain. perhaps pass the password? this would help.
# check if there are things missing.
def download_from_server(username, servername, src_path, dst_path, recursive=False):
    if recursive:
        subprocess.call(['scp', '-r', 
            remote_path(username, servername, src_path), dst_path])
    else:
        subprocess.call(['scp',
            remote_path(username, servername, src_path), dst_path])

def upload_to_server(username, servername, src_path, dst_path, recursive=False):
    if recursive:
        subprocess.call(['scp', '-r', src_path,
            remote_path(username, servername, dst_path)])
    else:
        subprocess.call(['scp', src_path,
            remote_path(username, servername, dst_path)])

# there are also questions about the existence of the path or not.
# also with the path manipulation functionality. as a last resort, I could 
# just delegate to the function that we call.

# TODO: the recursive functionality is not implemented. check if it is correct.
# needs additional checking for dst_path. this is not correct in general.
def download_from_url(url, dst_path, recursive=False):
    if recursive:
        raise NotImplementedError

    subprocess.call(['wget', url])
    filename = url.split('/')[-1]
    os.rename(filename, dst_path)

### parallelizing on a single machine
import psutil
import multiprocessing
import time

def mbs_process(pid):
    psutil_p = psutil.Process(pid)
    mem_p = psutil_p.memory_info()[0]
    
    return mem_p

def run_guarded_experiment(maxmemory_mbs, maxtime_secs, experiment_fn, **kwargs):
        start = time.time()
        
        p = multiprocessing.Process(target=experiment_fn, kwargs=kwargs)
        p.start()
        while p.is_alive():
            p.join(1.0)     
            try:
                mbs_p = mbs_process(p.pid)
                if mbs_p > maxmemory_mbs:
                    print "Limit of %0.2f MB exceeded. Terminating." % maxmemory_mbs
                    p.terminate()

                secs_p = time.time() - start
                if secs_p > maxtime_secs:
                    print "Limit of %0.2f secs exceeded. Terminating." % maxtime_secs
                    p.terminate()
            except psutil.NoSuchProcess:
                pass

# TODO: probably this works better with keyword args.
# NOTE: also, probably the way this is done is not the best way because how things 
# work.
def run_parallel_experiment(experiment_fn, iter_args):            
    ps = []                                 
    for args in iter_args:
        p = multiprocessing.Process(target=experiment_fn, args=args)
        p.start()
        ps.append(p)
    
    for p in ps:
        p.join()

### running remotely
import subprocess
import paramiko
import getpass

def get_password():
    return getpass.getpass()

def run_on_server(bash_command, servername, username=None, password=None,
        folderpath=None, wait_for_output=True, prompt_for_password=False):
    """SSH into a machine and runs a bash command there. Can specify a folder 
    to change directory into after logging in.
    """

    # getting credentials
    if username != None: 
        host = username + "@" + servername 
    else:
        host = servername
    
    if password == None and prompt_for_password:
        password = getpass.getpass()

    if not wait_for_output:
        bash_command = "nohup %s &" % bash_command

    if folderpath != None:
        bash_command = "cd %s && %s" % (folderpath, bash_command) 

    sess = paramiko.SSHClient()
    sess.load_system_host_keys()
    #ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    sess.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sess.connect(servername, username=username, password=password)
    stdin, stdout, stderr = sess.exec_command(bash_command)
    
    # depending if waiting for output or not.
    if wait_for_output:
        stdout_lines = stdout.readlines()
        stderr_lines = stderr.readlines()
    else:
        stdout_lines = None
        stderr_lines = None

    sess.close()
    return stdout_lines, stderr_lines

### running jobs on available computational nodes.
import inspect
import uuid

# tools for handling the lithium server, which has no scheduler.
def get_lithium_nodes():
    return {'gtx970' : ["dual-970-0-%d" % i for i in range(0, 13) ], 
            'gtx980' : ["quad-980-0-%d" % i for i in [0, 1, 2] ],
            'k40' : ["quad-k40-0-0", "dual-k40-0-1", "dual-k40-0-2"],
            'titan' : ["quad-titan-0-0"]
            }

def get_lithium_resource_availability(servername, username, password=None,
        abort_if_any_node_unavailable=True, 
        nodes_to_query=None):

    # prompting for password if asked about. (because lithium needs password)
    if password == None:
        password = getpass.getpass()

    # query all nodes if None
    if nodes_to_query == None:
        nodes_to_query = flatten( get_lithium_nodes() )

    # script to define the functions to get the available resources.
    cmd_lines = ['import psutil', 'import subprocess', 'import numpy as np', '']
    fns = [ convert_between_byte_units,
            cpus_total, memory_total, gpus_total, 
            cpus_free, memory_free,  gpus_free, gpus_free_ids]
    
    for fn in fns:
        cmd_lines += [line.rstrip('\n') 
            for line in inspect.getsourcelines(fn)[0] ]
        cmd_lines.append('')

    cmd_lines += ['print \'%d;%.2f;%d;%d;%.2f;%d;%s\' % ('
               'cpus_total(), memory_total(), gpus_total(), '
               'cpus_free(), memory_free(), gpus_free(), '
               '\' \'.join( map(str, gpus_free_ids()) ) )']
    py_cmd = "\n".join(cmd_lines)

    ks = ['cpus_total', 'mem_mbs_total', 'gpus_total', 
            'cpus_free', 'mem_mbs_free', 'gpus_free', 'free_gpu_ids']

    # run the command to query the information.
    nodes = get_lithium_nodes()
    write_script_cmd = 'echo \"%s\" > avail_resources.py' % py_cmd
    run_on_server(write_script_cmd, servername, username, password)

    # get it for each of the models
    node_to_resources = {}
    for host in nodes_to_query:
        cmd = 'ssh -T %s python avail_resources.py' % host
        stdout_lines, stderr_lines = run_on_server(
            cmd, servername, username, password)
        
        print stdout_lines, stderr_lines

        # if it did not fail.
        if len(stdout_lines) == 1:
            # extract the actual values from the command line.
            str_vs = stdout_lines[0].strip().split(';')
            assert len(str_vs) == 7
            print str_vs
            vs = [ fn( str_vs[i] ) 
                for (i, fn) in enumerate([int, float, int] * 2 + [str]) ]
            vs[-1] = [int(x) for x in vs[-1].split(' ') if x != '']

            d = create_dict(ks, vs)
            node_to_resources[host] = d
        else:
            assert not abort_if_any_node_unavailable
            node_to_resources[host] = None

    delete_script_cmd = 'rm avail_resources.py'
    run_on_server(delete_script_cmd, servername, username, password)
    return node_to_resources


# TODO: add functionality to check if the visible gpus are busy or not 
# and maybe terminate upon that event.
### this functionality is actually quite hard to get right.
# running on one of the compute nodes.
# NOTE: this function has minimum error checking.
def run_on_lithium_node(bash_command, node, servername, username, password=None, 
        visible_gpu_ids=None, folderpath=None, 
        wait_for_output=True, run_on_head_node=False):

    # check that node exists.
    assert node in flatten( get_lithium_nodes() )

    # prompting for password if asked about. (because lithium needs password)
    if password == None:
        password = getpass.getpass()

    # if no visilbe gpu are specified, it creates a list with nothing there.
    if visible_gpu_ids is None:
        visible_gpu_ids = []
    
    # creating the command to run remotely.
    gpu_cmd = 'export CUDA_VISIBLE_DEVICES=%s' % ",".join(map(str, visible_gpu_ids))
    if not run_on_head_node:
        cmd = "ssh -T %s \'%s && %s\'" % (node, gpu_cmd, bash_command)
    else:
        # NOTE: perhaps repetition could be improved here. also, probably head 
        # node does not have gpus.
        cmd = "%s && %s" % (gpu_cmd, bash_command)

    return run_on_server(cmd, **retrieve_values(locals(), 
        ['servername', 'username', 'password', 'folderpath', 'wait_for_output']))

# TODO: perhaps add something to run on all lithium node.

# NOTE: this may require adding some information to the server.
# NOTE: if any of the command waits for output, it will mean that 
# it is going to wait until completion of that command until doing the other 
# one.
# NOTE: as lithium does not have a scheduler, resource management has to be done
# manually. This one has to prompt twice.
# NOTE: the time budget right now does not do anything.

# TODO: password should be input upon running, perhaps.
# TODO: run on head node should not be true for these, because I 
# do all the headnode information here.
# TODO: perhaps add head node functionality to the node part, but the problem 
# is that I don't want to have those there.
# TODO: this is not very polished right now, but it should  be useful nonetheless.
class LithiumRunner:
    def __init__(self, servername, username, password=None, 
            only_run_if_can_run_all=True):
        self.servername = servername
        self.username = username
        self.password = password if password is not None else get_password()
        self.jobs = []

    def register(self, bash_command, num_cpus=1, num_gpus=0, 
        mem_budget=8.0, time_budget=60.0, mem_units='gb', time_units='m', 
        folderpath=None, wait_for_output=True, 
        require_gpu_types=None, require_nodes=None, run_on_head_node=False):

        # NOTE: this is not implemented for now.
        assert not run_on_head_node
        
        # should not specify both.
        assert require_gpu_types is None or require_nodes is None

        self.jobs.append( retrieve_values(locals(), 
            ['bash_command', 'num_cpus', 'num_gpus', 
            'mem_budget', 'time_budget', 'mem_units', 'time_units', 
            'folderpath', 'wait_for_output', 
            'require_gpu_types', 'require_nodes', 'run_on_head_node']) )

    def run(self, run_only_if_enough_resources_for_all=True):
        args = retrieve_values(vars(self), ['servername', 'username', 'password'])
        args['abort_if_any_node_unavailable'] = False

        # get the resource availability and filter out unavailable nodes.
        d = get_lithium_resource_availability( **args )
        d = { k : v for (k, v) in d.iteritems() if v is not None }
        
        g = get_lithium_nodes()

        # assignments to each of the registered jobs
        run_cfgs = []
        for x in self.jobs:
            if x['require_nodes'] is not None:
                req_nodes = x['require_nodes']  
            else: 
                req_nodes = d.keys()

            # based on the gpu type restriction.
            if x['require_gpu_types'] is not None:
                req_gpu_nodes = flatten(
                    retrieve_values(g, x['require_gpu_types']) )
            else:
                # NOTE: only consider the nodes that are available anyway.
                req_gpu_nodes = d.keys()
            
            # potentially available nodes to place this job.
            nodes = list( set(req_nodes).intersection(req_gpu_nodes) )
            assert len(nodes) > 0

            # greedy assigned to a node.
            assigned = False
            for n in nodes:
                r = d[n] 
                # if there are enough resources on the node, assign it to the 
                # job.
                if (( r['cpus_free'] >= x['num_cpus'] ) and 
                    ( r['gpus_free'] >= x['num_gpus'] ) and 
                    ( r['mem_mbs_free'] >= 
                        convert_between_byte_units( x['mem_budget'], 
                            src_units=x['mem_units'], dst_units='mb') ) ):  

                    # record information about where to run the job.
                    run_cfgs.append( {
                        'node' : n, 
                        'visible_gpu_ids' : r['free_gpu_ids'][ : x['num_gpus'] ]} )

                    # deduct the allocated resources from the available resources
                    # for that node.
                    r['cpus_free'] -= x['num_cpus']
                    r['gpus_free'] -= x['num_gpus']
                    r['mem_mbs_free'] -= convert_between_byte_units(
                        x['mem_budget'], src_units=x['mem_units'], dst_units='mb')
                    r['free_gpu_ids'] = r['free_gpu_ids'][ x['num_gpus'] : ]
                    # assigned = True
                    break
            
            # if not assigned, terminate without doing anything.
            if not assigned:
                run_cfgs.append( None )
                if run_only_if_enough_resources_for_all:
                    print ("Insufficient resources to satisfy"
                        " (cpus=%d, gpus=%d, mem=%0.3f%s)" % (
                        x['num_cpus'], x['num_gpus'], 
                        x['mem_budget'], x['mem_units'] ) )
                    return None

        # running the jobs that have a valid config.
        remaining_jobs = []
        outs = []
        for x, c in zip(self.jobs, run_cfgs):
            if c is None:
                remaining_jobs.append( x )
            else:
                out = run_on_lithium_node( **merge_dicts([
                    retrieve_values(vars(self),
                        ['servername', 'username', 'password'] ),
                    retrieve_values(x, 
                        ['bash_command', 'folderpath', 
                        'wait_for_output', 'run_on_head_node'] ),
                    retrieve_values(c, 
                        ['node', 'visible_gpu_ids'] ) ]) 
                )
                outs.append( out )

        # keep the jobs that 
        self.jobs = remaining_jobs
        return outs

# try something like a dry run to see that this is working.

# what happens if some node is unnavailable.
# TODO: I need to parse these results.
            
# there is also a question about what should be done in the case where there 
# are not many resources.

# something to register a command to run.

# as soon as a command is sent to the server, it is removed from the 
# list.
    
## TODO: do a lithium launcher that manages the resources. this makes 
# it easier.

# there is stuff that I need to append. for example, a lot of these are simple 
# to run. there is the syncing question of folders, which should be simple. 
# it is just a matter of  

    # depending on the requirements for running this task, different machines 
    # will be available. if you ask too many requirement, you may not have 
    # any available machines.

# be careful about running things in the background.
# the point is that perhaps I can run this command in 
# the background, which may actually work. that would be something interesting 
# to see if th 


# NOTE: there may exist problems due to race conditions, but this can be 
# solved later.

def run_on_matrix(bash_command, servername, username, password=None, 
        num_cpus=1, num_gpus=0, 
        mem_budget=8.0, time_budget=60.0, mem_units='gb', time_units='m', 
        folderpath=None, wait_for_output=True, 
        require_gpu_type=None, run_on_head_node=False, jobname=None):

    assert (not run_on_head_node) or num_gpus == 0
    assert require_gpu_type is None ### NOT IMPLEMENTED YET.

    # prompts for password if it has not been provided
    if password == None:
        password = getpass.getpass()

    script_cmd = "\n".join( ['#!/bin/bash', bash_command] )
    script_name = "run_%s.sh" % uuid.uuid4() 
    
    # either do the call using sbatch, or run directly on the head node.
    if not run_on_head_node:
        cmd_parts = [ 'sbatch', 
            '--cpus-per-task=%d' % num_cpus,
            '--gres=gpu:%d' % num_gpus,
            '--mem=%d' % convert_between_byte_units(mem_budget, 
                src_units=mem_units, dst_units='mb'),
            '--time=%d' % convert_between_time_units(time_budget, 
                time_units, dst_units='m') ]
        if jobname is not None:
            cmd_parts += ['--job-name=%s' % jobname] 
        cmd_parts += [script_name]

        run_script_cmd = ' '.join(cmd_parts)

    else:
        run_script_cmd = script_name    

    # actual command to run remotely
    remote_cmd = " && ".join( [
        "echo \'%s\' > %s" % (script_cmd, script_name), 
        "chmod +x %s" % script_name,
        run_script_cmd,  
        "rm %s" % script_name] )

    return run_on_server(remote_cmd, **retrieve_values(
        locals(), ['servername', 'username', 'password', 
            'folderpath', 'wait_for_output']) )

# TODO: needs to manage the different partitions.


# def run_on_bridges(bash_command, servername, username, password, num_cpus=1, num_gpus=0, password=None,
#         folderpath=None, wait_for_output=True, prompt_for_password=False,
#         require_gpu_type=None):
#     raise NotImplementedError
    # pass

# TODO: waiting between jobs. this is something that needs to be done.

### NOTE: I shoul dbe able to get a prompt easily from the command line.
# commands are going to be slightly different.

# NOTE: if some of these functionalities are not implemented do something.

# this one is identical to matrix.

# have a sync files from the server.

# NOTE: this is going to be done in the head node of the servers for now.
# NOTE: may return information about the different files.
# may do something with the verbose setting.

# should be able to do it locally too, I guess.

# only update those that are newer, or something like that.
# complain if they are different f

# only update the newer files. stuff like that. 
# this is most likely

# there is the nuance of whether I want it to be insight or to become 
# something.

# only update files that are newer on the receiver.

# stuff for keeping certain extensions and stuff like that.
# -R, --relative              use relative path names
# I don't get all the options, bu t a few of them should be enough.
# TODO: check if I need this one.

# TODO: this can be more sophisticated to make sure that I can run this 
# by just getting a subset of the files that are in the source directory.
# there is also questions about how does this interact with the other file
# management tools.

# NOTE: this might be useful but for the thing that I have in mind, these 
# are too many options.
# NOTE: this is not being used yet.
def rsync_options(
        recursive=True,
        preserve_source_metadata=True, 
        only_transfer_newer_files=False, 
        delete_files_on_destination_notexisting_on_source=False,
        delete_files_on_destination_nottransfered_from_source=False,
        dry_run_ie_do_not_transfer=False,
        verbose=True):
    
    opts = []
    if recursive:
        opts += ['--recursive']
    if preserve_source_metadata:
        opts += ['--archive']
    if only_transfer_newer_files:
        opts += ['--update']
    if delete_files_on_destination_notexisting_on_source:
        opts += ['--delete']
    if delete_files_on_destination_nottransfered_from_source:
        opts += ['--delete-excluded']
    if dry_run_ie_do_not_transfer:
        opts += ['--dry-run']
    if verbose:
        opts += ['--verbose']

    return opts                       

# there is compression stuff and other stuff.

# NOTE: the stuff above is useful, but it  is a little bit too much.
# only transfer newer.

# delete on destination and stuff like that. I think that these are too many 
# options.

# deletion is probably a smart move too...

# imagine that keeps stuff that should not been there. 
# can also, remove the folder and start from scratch
# this is kind of tricky to get right. for now, let us just assume that 

# there is the question about what kind of models can work.
# for example, I think that it is possible to talk abougt 

# the other deletion aspects may make sense.

# rsync opts

# do something directly with rsync.


# NOTE: that because this is rsync, you have to be careful about 
# the last /. perhaps add a condition that makes this easier to handle.
def sync_local_folder_from_local(src_folderpath, dst_folderpath, 
        only_transfer_newer_files=True):

    cmd = ['rsync', '--verbose', '--archive']
    
    # whether to only transfer newer files or not.
    if only_transfer_newer_files:
        cmd += ['--update']

    # add the backslash if it does not exist. this does the correct
    # thing with rsync.
    if src_folderpath[-1] != '/':
        src_folderpath = src_folderpath + '/'
    if dst_folderpath[-1] != '/':
        dst_folderpath = dst_folderpath + '/'

    cmd += [src_folderpath, dst_folderpath]
    out = subprocess.check_output(cmd)

    return out

# NOTE: will always prompt for password due to being the simplest approach.
# if a password is necessary , it will prompt for it.
# also, the dst_folderpath should already be created.
def sync_remote_folder_from_local(src_folderpath, dst_folderpath,
        servername, username=None, 
        only_transfer_newer_files=True):

    cmd = ['rsync', '--verbose', '--archive']
    
    # whether to only transfer newer files or not.
    if only_transfer_newer_files:
        cmd += ['--update']

    # remote path to the folder that we want to syncronize.
    if src_folderpath[-1] != '/':
        src_folderpath = src_folderpath + '/'
    if dst_folderpath[-1] != '/':
        dst_folderpath = dst_folderpath + '/'    
        
    dst_folderpath = "%s:%s" % (servername, dst_folderpath)
    if username is not None:
        dst_folderpath = "%s@%s" % (username, dst_folderpath)

    cmd += [src_folderpath, dst_folderpath]
    out = subprocess.check_output(cmd)
    return out

def sync_local_folder_from_remote(src_folderpath, dst_folderpath,
        servername, username=None, 
        only_transfer_newer_files=True):

    cmd = ['rsync', '--verbose', '--archive']

    # whether to only transfer newer files or not.
    if only_transfer_newer_files:
        cmd += ['--update']

    # remote path to the folder that we want to syncronize.
    if src_folderpath[-1] != '/':
        src_folderpath = src_folderpath + '/'
    if dst_folderpath[-1] != '/':
        dst_folderpath = dst_folderpath + '/'    
        
    src_folderpath = "%s:%s" % (servername, src_folderpath)
    if username is not None:
        src_folderpath = "%s@%s" % (username, src_folderpath)

    cmd += [src_folderpath, dst_folderpath]
    out = subprocess.check_output(cmd)
    return out

### TODO: add compress options. this is going to be interesting.

# there are remove tests for files to see if they exist.
# I wonder how this can be done with run on server. I think that it is good 
# enough.



# can always do it first and then syncing. 

# can use ssh for that.

# TODO: make sure that those things exist.




# there is the override and transfer everything, and remove everything.

# NOTE: syncing is not the right name for this.

    # verbose, wait for termination, just fork into the background.

# NOTE: the use case is mostly to transfer stuff that exists somewhere
# from stuff that does not exist.



### plotting 
import os
if "DISPLAY" not in os.environ or os.environ["DISPLAY"] == ':0.0':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: edit with retrieve_vars
class LinePlot:
    def __init__(self, title=None, xlabel=None, ylabel=None):
        self.data = []
        self.cfg = {'title' : title,
                    'xlabel' : xlabel,
                    'ylabel' : ylabel}
    
    def add_line(self, xs, ys, label=None, err=None):
        d = {"xs" : xs, 
             "ys" : ys, 
             "label" : label,
             "err" : err}
        self.data.append(d)

    def plot(self, show=True, fpath=None):
        f = plt.figure()
        for d in self.data:
             plt.errorbar(d['xs'], d['ys'], yerr=d['err'], label=d['label'])
            
        plt.title(self.cfg['title'])
        plt.xlabel(self.cfg['xlabel'])
        plt.ylabel(self.cfg['ylabel'])
        plt.legend(loc='best')

        if fpath != None:
            f.savefig(fpath, bbox_inches='tight')
        if show:
            f.show()
        return f

# TODO: I guess that they should be single figures.
class GridPlot:
    def __init__(self):
        pass
    
    def add_plot(self):
        pass
    
    def new_row(self):
        pass
    
    def plot(self, show=True, fpath=None):
        pass

### TODO: 
class BarPlot:
    def __init__(self):
        pass

# class ScatterPlot:
#     def __init__(self):
#         pass

#     self.cfg = {}

## TODO: perhaps make it easy to do references.
class GridPlot:
    def __init__(self):
        pass

# another type of graph that is common.
# which is.
    
# TODO: log plots vs non log plots. more properties to change.
# TODO: updatable figure.

# TODO: add histogram plo, and stuff like that.

### data to latex

# perhaps add hlines if needed lines and stuff like that. this can be done.
# NOTE: perhaps can have some structured representation for the model.
# TODO: add the latex header. make it multi column too.
def generate_latex_table(mat, num_places, row_labels=None, column_labels=None,
        bold_type=None, fpath=None, show=True):
    assert bold_type == None or (bold_type[0] in {'smallest', 'largest'} and 
        bold_type[1] in {'in_row', 'in_col', 'all'})
    assert row_labels == None or len(row_labels) == mat.shape[0] and ( 
        column_labels == None or len(column_labels) == mat.shape[1])
    
    # effective number of latex rows and cols
    num_rows = mat.shape[0]
    num_cols = mat.shape[1]
    if row_labels != None:
        num_rows += 1
    if column_labels != None:
        num_cols += 1

    # round data
    proc_mat = np.round(mat, num_places)
    
    # determine the bolded entries:
    bold_mat = np.zeros_like(mat, dtype='bool')
    if bold_type != None:
        if bold_type[0] == 'largest':
            aux_fn = np.argmax
        else:
            aux_fn = np.argmin
        
        # determine the bolded elements; if many conflict, bold them all.
        if bold_type[1] == 'in_row':
            idxs = aux_fn(proc_mat, axis=1)
            for i in xrange(mat.shape[0]):
                mval = proc_mat[i, idxs[i]]
                bold_mat[i, :] = (proc_mat[i, :] == mval)

        elif bold_type[1] == 'in_col':
            idxs = aux_fn(proc_mat, axis=0)
            for j in xrange(mat.shape[1]):
                mval = proc_mat[idxs[j], j]
                bold_mat[:, j] = (proc_mat[:, j] == mval)
        else:
            idx = aux_fn(proc_mat)
            for j in xrange(mat.shape[1]):
                mval = proc_mat[:][idx]
                bold_mat[:, j] = (proc_mat[:, j] == mval)

    # construct the strings for the data.
    data = np.zeros_like(mat, dtype=object)
    for (i, j) in iter_product(range(mat.shape[0]), range(mat.shape[1])):
        s = "%s" % proc_mat[i, j]
        if bold_mat[i, j]:
            s = "\\textbf{%s}" % s
        data[i, j] = s
    
    header = ''
    if column_labels != None:
        header = " & ".join(column_labels) + " \\\\ \n"
        if row_labels != None:
            header = "&" + header
    
    body = [" & ".join(data_row) + " \\\\ \n" for data_row in data]
    if row_labels != None:
        body = [lab + " & " + body_row for (lab, body_row) in zip(row_labels, body)]

    table = header + "".join(body)
    if fpath != None:
        with open(fpath, 'w') as f:
            f.write(table)
    if show:
        print table

### sequences, schedules, and counters (useful for training)
import numpy as np

# TODO: this sequence is not very useful.
class Sequence:
    def __init__(self):
        self.data = []
    
    def append(self, x):
        self.data.append(x)

# TODO: this can be done with the patience counter.
class PatienceRateSchedule:
    def __init__(self, rate_init, rate_mult, rate_patience,  
            rate_max=np.inf, rate_min=-np.inf, minimizing=True):
 
        assert (rate_patience > 0 and (rate_mult > 0.0 and rate_mult <= 1.0) and
            rate_min > 0.0 and rate_max >= rate_min) 
        
        self.rate_init = rate_init
        self.rate_mult = rate_mult
        self.rate_patience = rate_patience
        self.rate_max = rate_max
        self.rate_min = rate_min
        self.minimizing = minimizing
        
        # for keeping track of the learning rate updates
        self.counter = rate_patience
        self.prev_value = np.inf if minimizing else -np.inf
        self.cur_rate = rate_init

    def update(self, v):
        # if it improved, reset counter.
        if (self.minimizing and v < self.prev_value) or (
                (not self.minimizing) and v > self.prev_value) :
            self.counter = self.rate_patience
        else:
            self.counter -= 1
            if self.counter == 0:
                self.cur_rate *= self.rate_mult
                # rate truncation
                self.cur_rate = min(max(self.rate_min, self.cur_rate), self.rate_max)
                self.counter = self.rate_patience

        self.prev_value = min(v, self.prev_value) if self.minimizing else max(v, self.prev_value)

    def get_rate(self):
        return self.cur_rate

# TODO: write AdditiveRateSchedule; MultiplicativeRateSchedule.
class AddRateSchedule:
    def __init__(self, rate_init, rate_end, duration):
        
        assert rate_init > 0 and rate_end > 0 and duration > 0

        self.rate_init = rate_init
        self.rate_delta = (rate_init - rate_end) / float(duration)
        self.duration = duration
        
        self.num_steps = 0
        self.cur_rate = rate_init

    def update(self, v):
        assert self.num_steps < self.duration
        self.num_steps += 1

        self.cur_rate += self.rate_delta 

    def get_rate(self):
        return self.cur_rate

class MultRateSchedule:
    def __init__(self, rate_init, rate_mult):
        
        assert rate_init > 0 and rate_mult > 0

        self.rate_init = rate_init
        self.rate_mult = rate_mult

    def update(self, v):
        self.cur_rate *= self.rate_mult

    def get_rate(self):
        return self.cur_rate

class ConstantRateSchedule:
    def __init__(self, rate):
        self.rate = rate
    
    def update(self, v):
        pass
    
    def get_rate(self):
        return self.rate

class StepwiseRateSchedule:
    def __init__(self, rates, durations):
        assert len( rates ) == len( durations )

        self.schedule = PiecewiseSchedule( 
            [ConstantRateSchedule(r) for r in rates],
            durations)

    def update(self, v):
        pass
    
    def get_rate(self):
        return self.schedule.get_rate()

class PiecewiseSchedule:
    def __init__(self, schedules, durations):
       
        assert len(schedules) > 0 and len(schedules) == len(durations) and (
            all([d > 0 for d in durations]))

        self.schedules = schedules
        self.durations = durations

        self.num_steps = 0
        self.idx = 0

    def update(self, v):
        n = self.num_steps
        self.idx = 0
        for d in durations:
            n -= d
            if n < 0:
                break
            self.idx += 1
        
        self.schedules[self.idx]

    def get_rate(self):
        return self.schedules[self.idx].get_rate()

class PatienceCounter:
    def __init__(self, patience, init_val=None, minimizing=True, improv_thres=0.0):
        assert patience > 0 
        
        self.minimizing = minimizing
        self.improv_thres = improv_thres
        self.patience = patience
        self.counter = patience

        if init_val is not None:
            self.best = init_val
        else:
            if minimizing:
                self.best = np.inf
            else:
                self.best = -np.inf

    def update(self, v):
        assert self.counter > 0

        # if it improved, reset counter.
        if (self.minimizing and self.best - v > self.improv_thres) or (
            (not self.minimizing) and v - self.best > self.improv_thres) :

            self.counter = self.patience
        
        else:

            self.counter -= 1

        # update with the best seen so far.
        if self.minimizing:
            self.best = min(v, self.best)  
        
        else: 
            self.best = max(v, self.best)

    def has_stopped(self):
        return self.counter == 0

# NOTE: actually, state_dict needs a copy.
# example
# cond_fn = lambda old_x, x: old_x['acc'] < x['acc']
# save_fn = lambda x: tb.copy_update_dict(x, { 'model' : x['model'].save_dict() })
# load_fn = lambda x: x,
# for example, but it allows more complex functionality.
class Checkpoint:
    def __init__(self, initial_state, cond_fn, save_fn, load_fn):
        self.state = initial_state
        self.cond_fn = cond_fn
        self.save_fn = save_fn
        self.load_fn = load_fn
    
    def update(self, x):
        if self.cond_fn(self.state, x):
            self.state = self.save_fn(x)
    
    def get(self):
        return self.load_fn( self.state )

def get_best(eval_fns, minimize):
    best_i = None
    if minimize:
        best_v = np.inf
    else:
        best_v = -np.inf

    for i, fn in enumerate(eval_fns):
        v = fn()
        if minimize:
            if v < best_v:
                best_v = v
                best_i = i
        else:
            if v > best_v:
                best_v = v
                best_i = i
    
    return (best_i, best_v)

### for storing the data
class InMemoryDataset:
    """Wrapper around a dataset for iteration that allows cycling over the 
    dataset. 

    This functionality is especially useful for training. One can specify if 
    the data is to be shuffled at the end of each epoch. It is also possible
    to specify a transformation function to applied to the batch before
    being returned by next_batch.

    """
    
    def __init__(self, X, y, shuffle_at_epoch_begin, batch_transform_fn=None):
        if X.shape[0] != y.shape[0]:
            assert ValueError("X and y the same number of examples.")

        self.X = X
        self.y = y
        self.shuffle_at_epoch_begin = shuffle_at_epoch_begin
        self.batch_transform_fn = batch_transform_fn
        self.iter_i = 0

    def get_num_examples(self):
        return self.X.shape[0]

    def next_batch(self, batch_size):
        """Returns the next batch in the dataset. 

        If there are fewer that batch_size examples until the end
        of the epoch, next_batch returns only as many examples as there are 
        remaining in the epoch.

        """

        n = self.X.shape[0]
        i = self.iter_i

        # shuffling step.
        if i == 0 and self.shuffle_at_epoch_begin:
            inds = np.random.permutation(n)
            self.X = self.X[inds]
            self.y = self.y[inds]

        # getting the batch.
        eff_batch_size = min(batch_size, n - i)
        X_batch = self.X[i:i + eff_batch_size]
        y_batch = self.y[i:i + eff_batch_size]
        self.iter_i = (self.iter_i + eff_batch_size) % n

        # transform if a transform function was defined.
        if self.batch_transform_fn != None:
            X_batch_out, y_batch_out = self.batch_transform_fn(X_batch, y_batch)
        else:
            X_batch_out, y_batch_out = X_batch, y_batch

        return (X_batch_out, y_batch_out)

### simple randomness
import random
import numpy as np

def set_random_seed(x=None):
    np.random.seed(x)
    random.seed(x)

def uniform_sample_product(lst_lst_vals, num_samples):
    n = len(lst_lst_vals)
    components = []
    for i, lst in enumerate(lst_lst_vals):
        idxs = np.random.randint(len(lst), size=num_samples)
        components.append( [lst[j] for j in idxs] )

    samples = []
    for i in xrange(num_samples):
        samples.append([ components[j][i] for j in xrange(n) ])
    return samples


### TODO: not finished
def sample(xs, num_samples, with_replacement=True, p=None):
    if p == None:
        pass

### for simple NLP
def keep_short_sentences(sents, maxlen):
    return [s for s in sents if len(s) <= maxlen]
# TODO: change this for sequences maybe. simpler.

def count_tokens(sents):
    tk_to_cnt = {}
    for s in sents:
        for tk in s:
            if tk in tk_to_cnt:
                tk_to_cnt[tk] += 1
            else:
                tk_to_cnt[tk] = 1
    return tk_to_cnt

def keep_most_frequent_tokens(tk_to_cnt, num_tokens):
    top_items = sorted(tk_to_cnt.items(), 
        key=lambda x: x[1], reverse=True)[:num_tokens]
    return dict(top_items)

def remove_rare_tokens(tk_to_cnt, keep_thres):
    return {tk : c for (tk, c) in tk_to_cnt.iteritems() if c >= keep_thres}

def index_tokens(tokens, start_index=0):
    assert start_index >= 0
    num_tokens = len(tokens)
    tk_to_idx = dict(zip(tokens, range(start_index, num_tokens + start_index)))
    return tk_to_idx

def reverse_mapping(a_to_b):
    b_to_a = dict([(b, a) for (a, b) in a_to_b.iteritems()])
    return b_to_a

def reverse_nonunique_mapping(a_to_b):
    d = {}
    for (x, y) in a_to_b.iteritems():
        if y not in d:
            d[y] = []
        d[y].append(x)
    return d

def preprocess_sentence(sent, tk_to_idx, 
    unk_idx=-1, bos_idx=-1, eos_idx=-1, 
    bos_times=0, eos_times=0):
    """If no unk_idx is specified and there are tokens that are sentences that 
    have tokens that are not in the dictionary, it will throw an exception.
    it is also possible to look at padding using this type of functionality.
    """
    assert bos_idx == -1 or (bos_idx >= 0 and bos_times >= 0)  
    assert eos_idx == -1 or (eos_idx >= 0 and eos_times >= 0)  
    assert (unk_idx == -1 and all([tk in tk_to_idx for tk in sent])) or (unk_idx >= 0)

    proc_sent = []
    # adding the begin of sentence tokens
    if bos_idx != -1:
        proc_sent.extend([bos_idx] * bos_times)

    # preprocessing the sentence
    for tk in sent:
        if tk in tk_to_idx:
            idx = tk_to_idx[tk]
            proc_sent.append(idx)
        else:
            proc_sent.append(unk_idx)
            
    # adding the end of sentence token
    if eos_idx != -1:
        proc_sent.extend([eos_idx] * eos_times)

    return proc_sent

### simple featurization
class ProductFeaturizer:
    def __init__(self, fs):
        self.featurizers = fs
    
    def features(self, x):
        pass
    
    def dim(self):
        return sum( [f.dim() for f in self.featurizers] )

class HashFeaturizer:
    def __init__(self):
        pass

class OneHotFeaturizer:
    def __init__(self):
        pass

class BinarizationFeaturizer:
    def __init__(self):
        pass

# NOTE: certain things should be more compositional.

# TODO: perhaps add the is sparse information.

# what should be the objects.

# there is information about the unknown tokens and stuff like that.


# there are things that can be done through indexing.

# can featurize a set of objects, perhaps. 
# do it directly to each of the xs passed as 

# some auxiliary functions to create some of these.



# the construction of the dictionary can be done incrementally.

# TODO: a class that takes care of managing the domain space of the different 
# architectures.

# this should handle integer featurization, string featurization, float
# featurization
# 

# subset featurizers, and simple ways of featurizing models.

# features from side information.
# this is kind of a bipartite graph.
# each field may have different featurizers
# one field may have multiple featurizers
# one featurizers may be used in multiple fields
# one featurizer may be used across fields.
# featurizers may have a set of fields, and should be able to handle these 
# easily.

# features from history.
# features from different types of data. it should be easy to integrate.

# easy to featurize sets of elements, 

# handling gazeteers and stuff like that.
# it is essentially a dictionary. 
# there is stuff that I can do through

# I can also register functions that take something and compute something.
# and the feature is that. work around simple feature types.

# TODO: this is going to be quite interesting.

# come up with some reasonable interface for featurizers.
# TODO: have a few operations defined on featurizers.

# NOTE: that there may exist different featurizers.
# NOTE: I can have some form of function that guesses the types of elements.

# NOTE: for each csv field, I can register multiple features.

# NOTE that this is mainly for a row. what about for things with state.

### managing large scale experiments.
import itertools as it
import stat

def get_valid_name(folderpath, prefix):
    idx = 0
    while True:
        name = "%s%d" % (prefix, idx)
        path = join_paths([folderpath, name])
        if not path_exists(path):
            break
        else:
            idx += 1   
    return name

# generating the call lines for a code to main.
def generate_call_lines(main_filepath,
        argnames, argvals, 
        output_filepath=None, profile_filepath=None):

    sc_lines = ['python -u \\'] 
    # add the profiling instruction.
    if profile_filepath is not None:
        sc_lines += ['-m cProfile -o %s \\' % profile_filepath]
    sc_lines += ['%s \\' % main_filepath]
    # arguments for the call
    sc_lines += ['    --%s %s \\' % (k, v) 
        for k, v in it.izip(argnames[:-1], argvals[:-1])]
    # add the output redirection.
    if output_filepath is not None:
        sc_lines += ['    --%s %s \\' % (argnames[-1], argvals[-1]),
                    '    > %s 2>&1' % (output_filepath) ] 
    else:
        sc_lines += ['    --%s %s' % (argnames[-1], argvals[-1])] 
    return sc_lines

# all paths are relative to the current working directory or to entry folder path.
def create_run_script(main_filepath,
        argnames, argvals, script_filepath, 
        # entry_folderpath=None, 
        output_filepath=None, profile_filepath=None):
    
    sc_lines = [
        '#!/bin/bash',
        'set -e']
    # # change into the entry folder if provided.
    # if entry_folderpath is not None:
    #     sc_lines += ['cd %s' % entry_folderpath]
    # call the main function.
    sc_lines += generate_call_lines(**retrieve_values(locals(), 
        ['main_filepath', 'argnames', 'argvals', 
        'output_filepath', 'profile_filepath'])) 
    # change back to the previous folder if I change to some other folder.
    # if entry_folderpath is not None:
    #     sc_lines += ['cd -']
    write_textfile(script_filepath, sc_lines, with_newline=True)
    # give run permissions.
    st = os.stat(script_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(script_filepath, st.st_mode | exec_bits)

# NOTE: could be done more concisely with a for loop.
def create_runall_script(exp_folderpath):
    fo_names = list_folders(exp_folderpath, recursive=False, use_relative_paths=True)
    # print fo_names
    num_exps = len([n for n in fo_names if path_last_element(n).startswith('cfg') ])

    # creating the script.
    sc_lines = ['#!/bin/bash']
    sc_lines += [join_paths([exp_folderpath, "cfg%d" % i, 'run.sh']) 
        for i in xrange(num_exps)]

    # creating the run all script.
    out_filepath = join_paths([exp_folderpath, 'run.sh'])
    # print out_filepath
    write_textfile(out_filepath, sc_lines, with_newline=True)
    st = os.stat(out_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(out_filepath, st.st_mode | exec_bits)

# NOTE: for now, this relies on the fact that upon completion of an experiment
# a results.json file is generated.
def create_runall_script_with_parallelization(exp_folderpath):
    fo_names = list_folders(exp_folderpath, recursive=False, use_relative_paths=True)
    # print fo_names
    num_exps = len([n for n in fo_names if path_last_element(n).startswith('cfg') ])

    ind = ' ' * 4
    # creating the script.
    sc_lines = [
        '#!/bin/bash',
        'if [ "$#" -lt 0 ] && [ "$#" -gt 3 ]; then',
        '    echo "Usage: run.sh [worker_id num_workers] [--force-rerun]"',
        '    exit 1',
        'fi',
        'force_rerun=0',
        'if [ $# -eq 0 ] || [ $# -eq 1 ]; then',
        '    worker_id=0',
        '    num_workers=1',
        '    if [ $# -eq 1 ]; then',
        '        if [ "$1" != "--force-rerun" ]; then',
        '            echo "Usage: run.sh [worker_id num_workers] [--force-rerun]"',
        '            exit 1',
        '        else',
        '            force_rerun=1',
        '        fi',
        '    fi',
        'else',
        '    worker_id=$1',
        '    num_workers=$2',
        '    if [ $# -eq 3 ]; then',
        '        if [ "$3" != "--force-rerun" ]; then',
        '            echo "Usage: run.sh [worker_id num_workers] [--force-rerun]"',
        '            exit 1',
        '        else',
        '            force_rerun=1',
        '        fi',
        '    fi',
        'fi',
        'if [ $num_workers -le $worker_id ] || [ $worker_id -lt 0 ]; then',
        '    echo "Invalid call: requires 0 <= worker_id < num_workers."',
        '    exit 1',
        'fi'
        '',
        'num_exps=%d' % num_exps,
        'i=0',
        'while [ $i -lt $num_exps ]; do',
        '    if [ $(($i % $num_workers)) -eq $worker_id ]; then',
        '        if [ ! -f %s ] || [ $force_rerun -eq 1 ]; then' % join_paths(
                    [exp_folderpath, "cfg$i", 'results.json']),
        '            echo cfg$i',
        '            %s' % join_paths([exp_folderpath, "cfg$i", 'run.sh']),
        '        fi',
        '    fi',
        '    i=$(( $i + 1 ))',
        'done'
    ]
    # creating the run all script.
    out_filepath = join_paths([exp_folderpath, 'run.sh'])
    # print out_filepath
    write_textfile(out_filepath, sc_lines, with_newline=True)
    st = os.stat(out_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(out_filepath, st.st_mode | exec_bits)


# NOTE: not the perfect way of doing things, but it is a reasonable way for now.
# main_relfilepath is relative to the project folder path.
# entry_folderpath is the place it changes to before executing.
# if code and data folderpaths are provided, they are copied to the exp folder.
# all paths are relative I think that that is what makes most sense.
def create_experiment_folder(main_filepath,
        argnames, argvals_list, out_folderpath_argname, 
        exps_folderpath, readme, expname=None, 
        # entry_folderpath=None, 
        code_folderpath=None, 
        # data_folderpath=None,
        capture_output=False, profile_run=False):
    
    assert folder_exists(exps_folderpath)
    assert expname is None or (not path_exists(
        join_paths([exps_folderpath, expname])))
    # assert folder_exists(project_folderpath) and file_exists(join_paths([
    #     project_folderpath, main_relfilepath]))
    
    # TODO: add asserts for the part of the model that matters.

    # create the main folder where things for the experiment will be.
    if expname is None:    
        expname = get_valid_name(exps_folderpath, "exp")
    ex_folderpath = join_paths([exps_folderpath, expname])
    create_folder(ex_folderpath)

    # copy the code to the experiment folder.
    if code_folderpath is not None:
        code_foldername = path_last_element(code_folderpath)
        dst_code_fo = join_paths([ex_folderpath, code_foldername])

        copy_folder(code_folderpath, dst_code_fo, 
            ignore_hidden_files=True, ignore_hidden_folders=True, 
            ignore_file_exts=['.pyc'])

        # change main_filepath to use that new code.
        main_filepath = join_paths([ex_folderpath, main_filepath])

    # NOTE: no data copying for now, because it does not make much 
    # sense in most cases.
    data_folderpath = None ### TODO: remove later.
    # # copy the code to the experiment folder.    
    # if data_folderpath is not None:
    #     data_foldername = path_last_element(data_folderpath)
    #     dst_data_fo = join_paths([ex_folderpath, data_foldername])

    #     copy_folder(data_folderpath, dst_data_fo, 
    #         ignore_hidden_files=True, ignore_hidden_folders=True)

    # write the config for the experiment.
    write_jsonfile( 
        retrieve_values(locals(), [
        'main_filepath',
        'argnames', 'argvals_list', 'out_folderpath_argname', 
        'exps_folderpath', 'readme', 'expname',
        'code_folderpath', 'data_folderpath',
        'capture_output', 'profile_run']),
        join_paths([ex_folderpath, 'config.json']))

    # generate the executables for each configuration.
    argnames = list(argnames)
    argnames.append(out_folderpath_argname)
    for (i, vs) in enumerate(argvals_list): 
        cfg_folderpath = join_paths([ex_folderpath, "cfg%d" % i])
        create_folder(cfg_folderpath)

        # create the script
        argvals = list(vs)
        argvals.append(cfg_folderpath)
        call_args = retrieve_values(locals(), 
            ['argnames', 'argvals', 'main_filepath'])

        call_args['script_filepath'] = join_paths([cfg_folderpath, 'run.sh'])
        if capture_output:
            call_args['output_filepath'] = join_paths(
                [cfg_folderpath, 'output.txt'])
        if profile_run:
            call_args['profile_filepath'] = join_paths(
                [cfg_folderpath, 'profile.txt'])
        create_run_script(**call_args)

        # write a config file for each configuration
        write_jsonfile(
            create_dict(argnames, argvals), 
            join_paths([cfg_folderpath, 'config.json']))
    # create_runall_script(ex_folderpath)
    create_runall_script_with_parallelization(ex_folderpath)

    return ex_folderpath

def map_experiment_folder(exp_folderpath, fn):
    fo_paths = list_folders(exp_folderpath, recursive=False, use_relative_paths=False)
    num_exps = len([p for p in fo_paths if path_last_element(p).startswith('cfg') ])

    ps = []
    rs = []
    for i in xrange(num_exps):
        p = join_paths([exp_folderpath, 'cfg%d' % i])
        rs.append( fn(p) )
        ps.append( p )
    return (ps, rs)

def load_experiment_folder(exp_folderpath, json_filenames, 
    abort_if_notexists=True, only_load_if_all_exist=False):
    def _fn(cfg_path):
        ds = []
        for name in json_filenames:
            p = join_paths([cfg_path, name])

            if (not abort_if_notexists) and (not file_exists(p)):
                d = None
            # if abort if it does not exist, it is going to fail reading the file.
            else:
                d = read_jsonfile( p )
            ds.append( d )
        return ds
    
    (ps, rs) = map_experiment_folder(exp_folderpath, _fn)
    
    # filter only the ones that loaded successfully all files.
    if only_load_if_all_exist:
        proc_ps = []
        proc_rs = []
        for i in xrange( len(ps) ):
            if all( [x is not None for x in rs[i] ] ):
                proc_ps.append( ps[i] )
                proc_rs.append( rs[i] )
        (ps, rs) = (proc_ps, proc_rs)

    return (ps, rs)

### useful for generating configurations to run experiments.
import itertools 

def generate_config_args(d, ortho=False):
    ks = d.keys()
    if not ortho:
        vs_list = iter_product( [d[k] for k in ks] )
    else:
        vs_list = iter_ortho_all( [d[k] for k in ks], [0] * len(ks))
            
    argvals_list = []
    for vs in vs_list:
        proc_v = []
        for k, v in itertools.izip(ks, vs):
            if isinstance(k, tuple):

                # if it is iterable, unpack v 
                if isinstance(v, list) or isinstance(v, tuple):
                    assert len(k) == len(v)

                    proc_v.extend( v )

                # if it is not iterable, repeat a number of times equal 
                # to the size of the key.
                else:
                    proc_v.extend( [v] * len(k) )

            else:
                proc_v.append( v )

        argvals_list.append( proc_v )

    # unpacking if there are multiple tied argnames
    argnames = []
    for k in ks:
        if isinstance(k, tuple):
            argnames.extend( k )
        else: 
            argnames.append( k )
    
    # guarantee no repeats.
    assert len( set(argnames) ) == len( argnames ) 
    
    # resorting the tuples according to sorting permutation.
    idxs = argsort(argnames, [ lambda x: x ] )
    argnames = apply_permutation(argnames, idxs)
    argvals_list = [apply_permutation(vs, idxs) for vs in argvals_list]

    return (argnames, argvals_list)

# NOTE: this has been made a bit restrictive, but captures the main functionality
# that it is required to generate the experiments.
def copy_regroup_config_generator(d_gen, d_update):
    
    # all keys in the regrouping dictionary have to be in the original dict
    flat_ks = []     
    for k in d_update:
        if isinstance(k, tuple):
            assert all( [ ki in d_gen for ki in k ] )
            flat_ks.extend( k )
        else:
            assert k in d_gen
            flat_ks.append( k )

    # no tuple keys. NOTE: this can be relaxed by flattening, and reassigning,
    # but this is more work. 
    assert all( [ not isinstance(k, tuple) for k in d_gen] )
    # no keys that belong to multiple groups.
    assert len(flat_ks) == len( set( flat_ks ) ) 

    # regrouping of the dictionary.
    proc_d = dict( d_gen )
    
    for (k, v) in d_update.iteritems():
        
        # check that the dimensions are consistent.
        assert all( [ 
            ( ( not isinstance(vi, tuple) ) and ( not isinstance(vi, tuple) ) ) or 
            len( vi ) == len( k )
                for vi in v ] )

        if isinstance(k, tuple):

            # remove the original keys
            map(proc_d.pop, k)
            proc_d[ k ] = v

        else:

            proc_d[ k ] = v

    return proc_d

# TODO: syncing folders

### project manipulation
def create_project_folder(folderpath, project_name):
    fn = lambda xs: join_paths([folderpath, project_name] + xs)

    create_folder( fn( [] ) )
    # typical directories
    create_folder( fn( [project_name] ) )
    create_folder( fn( ["analyses"] ) )     
    create_folder( fn( ["data"] ) )
    create_folder( fn( ["experiments"] ) )
    create_folder( fn( ["notes"] ) )
    create_folder( fn( ["temp"] ) )

    # code files (in order): data, preprocessing, model definition, model training, 
    # model evaluation, main to generate the results with different relevant 
    # parameters, setting up different experiments, analyze the results and 
    # generate plots and tables.
    create_file( fn( [project_name, "__init__.py"] ) )
    create_file( fn( [project_name, "data.py"] ) )
    create_file( fn( [project_name, "preprocess.py"] ) )
    create_file( fn( [project_name, "model.py"] ) )    
    create_file( fn( [project_name, "train.py"] ) )
    create_file( fn( [project_name, "evaluate.py"] ) ) 
    create_file( fn( [project_name, "main.py"] ) )
    create_file( fn( [project_name, "experiment.py"] ) )
    create_file( fn( [project_name, "analyze.py"] ) )

    # add an empty script that can be used to download data.
    create_file( fn( ["data", "download_data.py"] ) )

    # common notes to keep around.
    create_file( fn( ["notes", "journal.txt"] ) )    
    create_file( fn( ["notes", "reading_list.txt"] ) )    
    create_file( fn( ["notes", "todos.txt"] ) )    

    # placeholders
    write_textfile( fn( ["experiments", "readme.txt"] ), 
        ["All experiments will be placed under this folder."] )

    write_textfile( fn( ["temp", "readme.txt"] ), 
        ["Here lie temporary files that are relevant or useful for the project "
        "but that are not kept under version control."] )

    write_textfile( fn( ["analyses", "readme.txt"] ), 
        ["Here lie files containing information extracted from the "
        "results of the experiments. Tables and plots are typical examples."] )

    # typical git ignore file.
    write_textfile( fn( [".gitignore"] ), 
        ["data", "experiments", "temp", "*.pyc", "*.pdf", "*.aux"] )
    
    subprocess.call("cd %s && git init && git add -f .gitignore * && "
        "git commit -a -m \"Initial commit for %s.\" && cd -" % ( 
            fn( [] ), project_name), shell=True)
    
# TODO: add options to directly initialize a git repo.


### simple tests and debugging that are often useful.
import itertools

def test_overfit(ds, overfit_val, overfit_key):
    ov_ds = []
    err_ds = []

    for d in ds:
        if d[overfit_key] == overfit_val:
            ov_ds.append( d )
        else:
            err_ds.append( d )
    
    return (ov_ds, err_ds)

def test_with_fn(ds, fn):
    ov_ds = []
    err_ds = []

    for d in ds:
        if fn( d ):
            good_ds.append( d )
        else:
            bad_ds.append( d )
    
    return (good_ds, bad_ds)

# assumes that eq_fn is transitive. can be used to check properties of
# a set of results.
def all_equivalent_with_fn(ds, eq_fn):
    for d1, d2 in itertools.izip(ds[:-1], ds[1:]):
        if not eq_fn(d1, d2):
            return False
    return True

def assert_length_consistency(xs_lst):
    assert len( set(map(len, xs_lst) ) ) == 1

    for i in xrange( len(xs_lst) ):
        assert set( [len(xs[i]) for xs in xs_lst] ) == 1


### utility functions for exploring mistakes in sequence data.
def mistakes_per_token(sents, gold_tags, pred_tags):
    pass

def mistakes_per_tag(sents, gold_tags, pred_tags):
    pass

def mistakes_per_sentence(sents, gold_tags, pred_tags):
    pass

def metric_per_sentence(sents, gold_tags, pred_tags, fn):
    pass

# NOTE: should I sort the sentences or just use them as they are.


# NOTE: this can be done differently and have options that determine if something 
# is done or not.
    # TODO: changes into the folder that I care about, runs something and 
    # returns to where I was before.
    # subprocess.call(["cd && "])

# TODO: I think that it would be nice to have some utility scripts that can 
# be directly added to the model.

# for example, scripts to download the data that you care about. 
# git ignore.

### string manipulation
def wrap_strings(s):
    return ', '.join(['\'' + a.strip() + '\'' for a in s.split(',')])

# string manipulation.

# TODO: something to split on spaces 
# or strip on spaces. and stuff

# join strings by separator.

# it is very useful to manage the output in such a way that it is easy 
# to send things to a file, as it is easy to send them to the terminal.
# 

# callibration for running things in the cpu.

# if there are multiple things to be loaded, it is best to just do one.
# or something like that.

# it would be nice if merging the keys would preserve some form of 
# ordering in the models.

# TODO: perhaps additional information that allows to run things on matrix 
# directly by syncing folders and launching threads.

# compress path? this is something that I can do conveniently.

# run on the server with multiple gpus and multiple experiments.
# easiest. split into multiple experiments.

# there is also the question about launching things in parallel, and 
# how can I go between Python and bash.

# NOTE: some of these things will not account for recursion in directory copying,
# not even I think that they should.

# send email functionality, or notification functionality. 
# sort of a report of progress made, how many need to be done more and stuff like 
# that.

### load all configs and stuff like that.
# they should be ordered.
# or also, return the names of the folders.
# it is much simpler, I think.

# stuff to deal with embeddings in pytorch.

# connecting two computational graphs in tensorflow.

# TODO: some functionality will have to change to reaccomodate
# the fact that it may change slightly.

### profiling


# the maximum I can do is two dimensional slices.

# get all configs.
# # TODO: add information for running this in parallel or with some other tweaks.
# def run_experiment_folder(exp_folderpath, ):
#     scripts = [p for p in list_files(exp_folderpath, recursive=True) 
#         if path_last_element(p) == 'run.sh'][:5]

#     print scripts
#     print len(scripts)
#     for s in scripts:
#         subprocess.call([s])
    
    ### call each one of them, it can also be done recursively.

# how to identify, them
# plot the top something models and see 
# problematic to represent when they have 30 numbers associated to them.
# pick up to 7 representative numbers and use those.

# top 10 plots for some of the most important ones.

# TODO: <IMPORTANT> standardization of the files generated by the experiment,
# and some functionality to process them.

### TODO: upload to server should compress the files first and decompress them 
# there.

# may comment out if not needed.

# for boilerplate editing

# add these. more explicitly.
# TODO: add file_name()
# TODO: file_folder
# TODO: folder_folder()
# TODO: folder_name()

# add more information for the experiments folder.
# need a simple way of loading a bunch of dictionaries.

# there is the question of it is rerun or dry run.

# basically, each on is a different job.
# I can base on the experiment folder to 
# with very small changes, I can use the data 
# data and the other one.
# works as long as paths are relative.

# this model kind of suggests to included more information about the script 
# that we want to run and use it later.

## NOTE: can add things in such a way that it is possible to run it
# on the same folder. in the sense that it is going to change folders
# to the right place.

# I think that I can get some of these variations by combining various iterators.

# a lot of experiments. I will try just to use some of the 
# experiments.

# running a lot of different configurations. 
# there are questions about the different models.
# 


# TODO: need to add condition to the folder creation that will make it 
# work better. like, if to create the whole folder structure or not.
# that would be nice.

# needs to create all the directories.

# needs to create the directories first.

# assume that the references are with respect to this call, so I can map them 
# accordingly

# stuff to kill all jobs. that can be done later.
# <IMPORTANT> this is important for matrix.

# NOTE: this may have a lot more options about 
def run_experiment_folder(folderpath):
    pass

# there needs to exist some information about running the experiments. can 
# be done through the research toolbox.

# assumes the code/ data 
# assumes that the relative paths are correct for running the experiment 
# from the current directory.

# this can be done much better, but for now, only use it in the no copying case.
# eventually, copy code, copy data depending on what we want to get out of it.
# it is going to be called from the right place, therefore there is no problem.


# even just upload the experiment folder is enough.

# TODO: stuff for concatenation. 

################################### TODO ###################################
# ### simple pytorch manipulation
# NOTE: this needs to be improved.
# import torch

# # TODO: with save and load dicts.
# def pytorch_save_model(model, fpath):
#     pass

# def pytorch_load_model(fpath):
#     pass



# have to hide the model for this.

# time utils

# tensorflow manipulation? maybe. installation of pytorch in some servers.

# call retrieve_values with locals()

# massive profiles.

# TODO:
def wait(x, units='s'):
    pass
    # do something simple.

# TODO: do a function to collapse a dictionary. only works for 
# key value pairs.

# add the ignore hidden files to the iterator.

# stuff to read models from the file, but serialized.

# stuff for checkpointing. resuming the model. what can be done.
# stuff for running multiple jobs easily.

# stuff for working with PyTorch.

# along with some interfaces.

# keep the most probable words.

# some of these are training tools. 
# this makes working with it easier.

### dealing with text.

# some default parameters for certain things.
# this is also useful to keep around.

# check if I can get the current process pid.

# managing some experimental folders and what not.

# will need to mask out certain gpus.

# probably, it is worth to do some name changing and stuff.

# these are mostly for the servers for which I have access.

# may need only the ones that are free.

# perhaps have something where it can be conditional on some information 
# about the folder to look at.
# If it is directly. it should be recursive, and have some 
# form of condition. I need to check into this.

# information on what gpu it was ran on. 
# check if some interfacing with sacred can be done.

# call these these functions somewhere else. 
# this should be possible.

# do some well defined units;
# have some well defined server names.

# creates the wrappers automatically.

# needs to be updated.

# folder creation and folder management.

# mostly for reading and writing, since I think it is always the same.
# I think that it should get a single figure for it.
# depending on what you pass, you should be able to get what you want.


# I don't think that I will need it.

# simple interfaces for models.

# for preprocessing
# for loading data
# for models
# for evaluation
# for visualization.

# do some more error checking for the writing the dictionaries to disk.
# like, no new lines

# make them strings, for ease of use.

# TODO: tools for managing a log folder automatically

# name experiments; if not, just have a single folder where everything is there. 
# sort by some date or id.

# in some cases, operate directly on files.
# take a list of files, and return a csv.

# wrap certain metrics to make them easy to use.

# prefer passing a dictionary, and then unpacking.

# something for creating paths out of components

# TODO: stuff for using Tensorflow and PyTorch and Tensorboard.

# better dictionary indexing and stuff like that.

# that can be done in a more interesting way.

# there are probably some good ways of getting the configurations.
# ideally, I think that it should be possible to sample configurations 
# uniformly.


# for tied configurations.
# basically, in cases where we want special conbinations.

# gives back, argnames, argvals.


# NOTE: there is probably something analogous that can be done for iterators.


# dictionary aggregate, test what are the value that each element has 
# it only makes sense to have such a dictionary for extra error checking.

# also, iteration over the folders that satisfy some characteristics, 
# like having certain values for certain parameters.

# also options about if there are two with the same parameters,
# I can always get the latest one.

# the experiment folder should be self-contained enough to just be copied to some
# place and be run there.
# resarch toolbox would be accessible through that folder,

# "import research_toolbox as tb; tb.run_experiment_folder("exp0", )

# these iterations cycles are simple, but take some time to figure out.

# depending on the dimensions.
# checking these configurations is interesting.

# do something that goes directly from DeepArchitect to a figure.

# something between interface between DeepArchictect and a Figure.

# add common ways of building a command line arguments.

# add common arguments and stuff like that.
# for learning rates, and step size reduction.

# for dealing with subsets, it is important to work with the model.

# do some structures for keeping track of things. easily.
# perhaps just passing a dictionary, and appending stuff.
# dealing with summary taking it is difficult.

# files to write things in the CONLL-format 

# I'm making wide use of Nones. I think that that is fine.
# keep this standard, in the sense that if an option is not used,
# it is set to None.

# stuff to make or abort to create a directory.

# iterate over all files matching a pattern. that can be done easily with re.

# simple scripts for working with video and images.

# the whole load and preprocessing part can be done efficiently, I think 
# I think that assuming one example per iterable is the right way of going 
# about this. also, consider that most effort is not in load or preprocessing
# the data, so we can more sloppy about it.
    

# NOTE: there are also other iterators that are interesting, but for now,
# I'm not using them. like change of all pairs and stuff like that.

# NOTE: numpy array vs lists. how to deal with these things.

# do some random experiment, that allows then to do a correction,
# for example, given 

# maybe passes on a sequence of tests that are defined over the 
# experiments. 
# this means that probably there is some loading of experiments 
# that is then used.

# set_seed
# and stuff like that. 
# it is more of an easy way of doing things that requires me to think less.

# copy directory is important
# as it is copy file.

# copy directory. all these can be done under the ignore if it exists 
# or something like that. I have to think about what is the right way of 
# going about it.

# create these summaries, which make things more interesting.

# managing these hyperparameter configurations is tedious. there is 
# probably a better way that does not require so many low level manipulations.

# problem. something in other languages will not work as well.
# does not matter, can still do most of it in Python.

# exposing the dataset. soem form of list or something like that.
# doing that, I think that it is possible to implement some common 
# transformations.

# can do some form of automatic collection, that gets everything into a file.
# the creation of these configuration dictionaries is really tedious.

# TODO: something for profiling a function or script.
# open the results very easily.

# TODO: add typical optimization strategies, such as how to do the 
# step size and how to reduce it. to the extent possible, make it 
# work on tensorflow and pytorch.

# make it easy to run any command remotely as it would locally.
# remote machines should be equally easy to use.

# look the DeepArchitect experiments to see what was done there.

# also, a lot of file based communication, like, you can generate the 
# output directly to a file.s

# add more restrictions in the keys that are valid for the dictionary
# I don't think that arbitrary strings are nice.

# another way of doing things is to write the names and capture 
# the variables from the local context, that way, I only have to write the names, 
# i.e., it becomes less tedious.

# functions to send emails upon termination. that is something interesting,
# and perhaps with some results of the analysis.

# also, it is important to keep things reproducible with respect to a model.
# some identification for running some experiment.

# plotting is also probably an interesting idea.

# TODO: logging can be done through printing, but there is also
# the question about append the stdin and stderr to 
# the file or terminal.

# also, potentially add some common models

# have an easy way of generating all configurations, but also, orthogonal 
# configurations.

# generate most things programmatically.

# some things can be done programmatically by using closures.
# to capture some arguments.

# also, given a bunch of dictionaries with the same keys, or perhaps 
# a few different keys, there are ways of working with the model directly.

# can have a few of them that go directly to a file.
# writes the configurations directly to a file.

# experiment id or something like that. 
# just the time and the commit.

# also, do things for a single row.

# get things out of a dictionary. I think that it should also be possible 
# call a function with the arguments in a dictionary.

# just the subset that it requires, I think that it is fine.
# it would be nice to pass extra stuff without being needed, but still 
# wouldn't have the model complain.

# important to have these auxiliary functions.

# probably those iterators would like to have a dictionary or a tuple
# that would be nice.

# essentially, question about using these paths, vs using something that is more 
# along the lines of variable arguments.

# working on the toolbox is very reusable.

# more like aggregated summary and stuff like that.

# especially the functions that take a large number of arguments are easier 
# to get them working.

# check if PyTorch keeps sparse matrices too? probably not
# Why do I care?

# stuff that is task based, like what wrappers and loaders, and preprocessors
# and stuff like that.

# stuff for image recognition, image captioning, visual question answering

# all these setups still stand.

# probably has to interact well with dictionaries. for example, I could 
# pass an arbitrary dictionary and it would still work.
# also should depend on the sequence, 

# and it may return a dictionary or a tuple. that seems reasonable enough.

# manipulation of text files is actually quite important to make this practical.
# tabs are not very nice to read.

# NOTE: going back and forth using these dictionaries is clearly something that 
# would be interesting to do easily.

# some capture of outputs to files.

# something to sample some number uniformly at random from the cross 
# product.
# it is simple to construct the list from the iterator

# better do at random and then inspect factors of variation.
# the most important thing is probably time and space.

# use the principles about being easy to try a new data set. 
# try a new method, try a new evaluation procedure
# try a model
# this is perhaps hard to do.
# maybe it is possible to develop functions that use that other functions 
# and sort of do that.
# like fit_functions.

# communication and returning stuff can be done through dictionaries and stuff like 
# that.

# logging is going to be done through printing.

# either returns one of these dictionaries or just changes it.

# be careful about managing paths. use the last one if possible, t

# stuff for string manipulation

# the summaries are important to run the experiments.

# simple featurizers for text, images, and other stuff.

# dictionaries are basically the basic unit into the model.

# another important aspect is the toolbox aspect, where you 
# things are more consistent that elsewhere

# creating command-line argument commands easily

# TODO: more stuff to manage the dictionaries.

### another important aspect.
# for creating interfaces.
# actual behavior of what to do with the values of the arguments
# is left for the for the program.
# use the values in 

# may specify the type or not.

# generating multiple related configs easily.
# perhaps by merging or something like that.
# and then taking the product.

# NOTE: may there is a get parser or something like that.

# easy to add arguments and stuff like that. what 
# cannot be done, do it by returning the parser.

# may be more convenient inside the shell.

# # get the output in the form of a file. that is good.

# get all available machines 
# TODO: this may be interesting to do at the model.

# this is especially useful from the head node, where you don't have to 
# login into machines.

# getting remote out and stuff like that.

# run on cpu and stuff like that.
# depending on the model.

# can also, run locally, but not be affected by the result.

# it works, it just works.

### some simple functions to read and write.

# get current commit of the model.

# stuff to handle dates.

# if it is just a matter of tiling the graph, it can be done.

# can either do it separately r 

# TODO: add stuff for easy hyperparameter tuning.
# it is easy to make the code break.
# also, check some of the simple stack traces up front.


# GridPlots and reference plots.

# all factors of variation.

# class ReferencePlot:
#     pass
# the goal is never have to leave the command line.

# scatter plots too. add these.

# folder creation

# TODO: generating a plot for a graph.
# TODO: generating code for a tree in tikz.

### for going from data to latex

# NOTE: no scientific notation until now.
# possibilities to add, scientific notation.
# to determine this, and what not.

# probably in the printing information, I should allows to choose the units
# TODO: the table generation part was not working OK yet.
# some problems, with bolding the elements 

# perhaps generate the table directly to a file.
# this works best and avoids cluttering the latex.

# maybe have other sequences by doing a comma separated list.
# mostly for single elements.
# used for setting up experiments.

### TODO: perhaps make this more expressive.

# all should be strings, just for ease of use.

# information about the single run can be generated somehow.
# maybe not in the results, but in some form of log.

# ignoring hidden folders is different than ignoring hidden files.
# add these two different options.

# perhaps ignoring hidden files vs ignoring hidden folders make more sense.

# can I do anything with remote paths to run things. I think that it may 
# be simpler to run models directly using other information.

# if no destination path, in the download, it assumes the current folder.
# TODO: always give the right download resulting directory.

# probably more error checking with respect to the path.

# TODO: perhaps will need to add ways of formating the json.

# NOTE: add support for nested columns.

# add support for grids of images.

# DO a conversion from a dict to an ordered dict
# TODO: something that makes it easy to generate different graphs.

# stuff for profiling and debugging.

# should be easy to compute statistics for all elements in the dataset.

# also, adopt a more functional paradigm for dealing with many things.
# all things that I can't do, will be done in way  


# using the correct definition is important. it will make things more consistent.

# even for simpler stuff where from the definition it is clear what is the 
# meaning of it, I think that it should be done.

# don't assume more than numpy.
# or always do it through lists to the extent possible.

# use full names in the functions to the extent possible

# use something really simple, in cases where it is very clear.

### this one can be used for the variables.

# json is the way to go to generate these files.

# decoupling how much time is it going to take. I think that this is going to 
# be simple.

### NOTE: this can be done both for the object and something else.

# it they exist, abort. have flag for this. for many cases, it make sense.

# important enough to distinguish between files and folders.

# perhaps better to utf encoding for sure.
# or maybe just ignore.

# for download, it also makes sense to distinguish between file and folder.

# for configfiles, it would be convenient.

# there is code that is not nice in terms of consistency of abstractions.
# needs to be improved.

# managing the data folder part of the model. how to manage folders with 
# data. perhaps register.

# add some stuff to have relative vs direct directories.
# get something about the current path.
# get information about hte working directory.

# running different configurations may correspond to different results.

# useful to have a dictionary that changes slightly. simpler than keeping 
# all the argumemnts around.
# useful when there are many arguments.

# run the model in many differetn 

# transverse all folders in search of results files and stuff like that.
# I think path manipulation makes a lot more sense with good tools.
# for total time and stuff like that.

# this can be useful to having stuff setup.

# stuff to pull all the log files from a server.
# stuff to easily generate multiple paths to ease boilerplate generation.
# for example, for generating long list of files to run in.

# for example, read a file that is named similarly from 
# all the elements of a folder. that file should be a dictionary.
# the question there is that it should probably json for ease of 
# use.

# apply a function to each of the elements, or may be not.

# that is an interesting way of getting the results from the model.

# can capture all the argumentss, I guess, which means that 
# that I can use the information.

# next thing: I need to make this more consistent.

# TODO: very important, check the use of the models that I need to

# merge multiple parsers.

# make it possible to merge multiple command ilne arguments.

# stuff to append. stuff to keep an average 
# and stuff like that.
# stuff to do both.

### all files matching a certain pattern. 
# 

# function to filter out. 

# doing the same thing in multiple machines.

# more stuff to deal with how to iterate over directory folders.

# can do a lot more for registering stuff.

# needs some more tools for managing directories.

# needs to improve stuff for reproducibility.

# needs to improve stuff for reproducibility.
# generating rerun scripts assuming data is in a certain place. 

# stuff for managing experiment folders.

# default configurations for running all, and running the ones we need.

# all structure to include in the generation script in there.
# this could be the general structure of a repo. 

# generate simple scripts for it. they should be able to get the absolute paths
# to the data.

# passing the data directory is good form for running the code.

# also, interactive filtering, perhaps with a dictionary, or something 
# like that. 

# running them from the command line makes more sense.

# basically, each task is calling the model some number of times 

# need to have a way of calling some python command on the command line 
# there and returning the results.

# this will return the results 

# some python command from some lib, with some addional imports.
# this will be the case. which can actually be quite long, 
# that would be quite useful actually.

# stuff for managing experiments necessary.

# add some simple function to retrieve information from the model.
# like time and other stuff.

# for directly generating experiments from main and stuff like that.

# get paths to the things that the model needs. I think that it is a better way of
# accomplishing this.

# analysis is typically single machine. output goes to something.

# copy_code.

# code in the experiment.
# stuff like that.

# for getting only files at some level

# TODO: creation of synthetic data.

# do the groupby in dictionaries
# and the summaries.

# should be simple to use the file iterator at a high level.
# produce the scripts to run the experiment.

# decide on the folder structure for the experiments... maybe have a few
# this is for setting up the experiments.

# I'm making a clean distinction betwenn the files that I want to run and 
# the files that I want to get.

# stuff to process some csv files.
# stuff to manipulate dates or to sort according to date. 
# perhaps create datetime objects from the strings.

# the question is sorting and organizing by the last run.
# it is also a question of 

# will probably need to set visability for certain variables first.

# the goal is to run these things manually, without having to go to the 
# server.

# most of the computational effort is somewhere else, so it is perfectly fine 
# to have most of the computation be done somewhere else.

# have a memory folder, it is enough to look for things. 
# can create experiments easily.

# having a main.
# it is possible to generate an experiment very easily using the file.
# and some options for keeping track of information.

# this can be done to run the code very easily.
# taking the output folder is crucial for success.

# if the code is there, there shouldn't exist really a problem with getting it to 
# work.

# this may have a lot of options that we want to do to get things to 
# work. maybe there is some prefix and suffix script that has to be ran for these
# models. this can be passed somewhere.

### TODO: other manipulation with the experiment folder.
# that can be done.

# it should copy the code without git and stuff. actually, it may be worth 
# to keep git.

# there are folders and subfolders. 

# may exist data_folderpath or something, but I would rather pass that 
# somewhere else. also, maybe need to change to some folder.

# TODO: probably do something to add requirements to the code. 
# this would be something interesting.

# run one of these folders on the server is actually quite interesting.
# I just have to say run folder, but only every some experiment.
# that is what I need.

# operating slurm form the command line.

# this is going to be important for matrix and the other ones.

# login
# change to the right folder
# call the correct function to run the experiment
# (another fucntion to get the results.)

# it really depends on what folder it should work 
# with. there is a question about what kind of running stuff will it work 
# with. it is really tricky to get it to run on multiple machines.

# it always refer within the data folder. this is the right thing 
# to do.

# create an experiment folder that is sort of on par.

# also, regardless of the path used to run the things, it should 
# always do the right thing..

# available memory per gpu and stuff like that.

# probably I should have some way of asking for resources.

# maybe wrap in a function.
# gpu machines in lithium.

### will need some configs for slurm. how to set things up.

# how to work with long running jobs and stuff like that.
# I need some status files for experiment folders.

# completing the task, would mean that the model, 

# for periodic saving things. 

# just need some path, the rest is agnostic to that path.
# just needs to know the path and that is it.

# simple jobs for now. and I will be happy.
# multiple runs for the project.

# readme json with other information

# may just find a new name.

# always ask for a name.

# simple naming conventions
# typical folder would be experiment.

# can be local or it can be global
# run the folder.

# have some tools for padding.
# it may have a way of doing reentry, or it may simply try different models.

### NOTE: I need stuff to get the configs out of the folder of the model.
# what else can we get from it.

# generates a run script and stuff like that.

# no defaults in the experiments.

# copy folder, but with current code. something like  that.

# to manipulate this folder, it is necessary to do the configs.

# programs need to be prepared for reentry.

# about experiment continuity and reentry.
# that needs to be done with the toolbox.

# look for files of a certain type.

#### ****************
# there are questions upon the creation of the model
# run.sh all (run fraction, )  
# out.txt
# config.json
# results.json
# .... (other files) ....
# main has to be prepared for reentry, that is it.
# and files have to be appended.

# needs easy way of making the code run directly. never know anything,
# for example, how does it know that it has been finished or not.

# do it all in terms of relative paths.
# or perhaps map them to non-relative paths.

# can add more configurations to the experiments folder.

# assumes that the config carries all the necessary information.

# assumes that the model knows what to do with respect to the given data path.

# it is a question of reentry and checkpoints and stuff. it runs 
# until it does not need to run anymore.
# perhaps keep the model.

# stuff that I can fork off with sbatch or something.

# get a subset of files from the server.
# that is quite boring to get all the information from the files.

# there is stuff about making parents and things like that.

# temp, out and other things that are more temporary and for which your command
# does not care too much.

# either all or a fraction.

# working with path fragments and stuff like that.

# commit on the structure.

# reading the results is going to be custom, but should be simple.

# the should exist download tools that mimic the directory structure, but only 
# get some of the files.
# needs to add some more information to the model.

# --parallel
# does one after the other, or something like that.

# list of modules to load and stuff like that. 
# I think that the experiment was done for the machine.
# after having all those things, there is no need of the research toolbox.

# TODO: I need stuff for GPU programs.

# relative paths are better.
# I don't see a reason to use absolute paths.
# works better in the case of copying to the server.

# the other aspect is visualization.
# that is quite important after you have the results.

# most commands are going to be submitted through the command line.
# it would be convenient to run the experiment creation in the model.
# that is something while the other part of the mdooe l 

# copy the results of the model to my persistent space.

# timing based on my iteration.
# that is something that is quite interesting.

# the fast config to run locally.
# I think that it would make sense to run those experiments.
# it should probably have some form of configuration itself that 
# is useful in reading the model.

# what can I do with that model. this is actually quite interesting. 

# NOTE: type of functions that can be considered to explore 

# for gpu programs and stuff like that
# maybe I could add options which are common in generating the bash scripts 
# such as what is the model doing and stuff like that.

# NOTE: it does not have to be directly this part of the model.
# it has to be some form of the model that should work directly.

### I will need to do something for working with this type of model
# in C or something like that. the question is that I need a few functions 
# to work with the model. 

# number of jobs. they can be ran in parallel, but they can also be ran in series.
# so why split. to have more variability in the results before getting 
# anything. 

# slurm remote runs, how to do it. how can I get it to work.
# remote slurm vs other type of slurm.

# the tools are important in getting the right amount of memory from the 
# model. for example, I model it quite easily. then, I can ask for 
# two times the necessary memory.

# do stuff with the scheduler.

# featurization would be convenient. 
# like some form of adding features. maybe, as a named dictionary 
# or positionally.
# these specifications are really boring, but they are necessary.
# there is really not much to do here.

# can be a function from the toolbox. 
# just creates the path as I want it.

# TODO: add the stuff for graph viz in both pytorch and tensorflow.
# the problem is that there are things that I need to care about.

# graph visualization can be crucial for understanding what is happening 
# in the model. it should be possible to wrap the model easily.

# using hooks to access intermediate gradients.

# TODO: can introduce some easy ways of testing the program? 
# it is hard to do this in general. perhaps in some narrow domain specific way 
# it is possible.

# experiments with sub experiments.

# it is better to run experiments from a single place, using the toolbox,
# and the config. 
# it seems annoying to have the config and the other ones.
# can lead to inconsistencies.

# allow for sub experiments.

# TODO: add support for nested experiments.
# I think that this is possible simply by having the right thing.
# these processes can be simple.

# example, a config that runs some number of other configs for some other 
# experiment.

# do things for the exploration of csv files. NOTE: can be done through pandas.
# this is going to be very handy to explore the results.

# easy way of fixing some of the arguments.

# NOTE: some of the stuff for connecting graphs in tensorflow.

# TODO: once it grows to a certain range, it will be useful to create toolbox 
# files mimicking some of the evaluation metrics that we looked at.
# this makes sense only after reaching a certain number of things to 
# do, or having libraries that are not guaranteed to exist in the model.

# TODO: for debugging, I can add information about the calling stack or 
# the function that is calling some methods. this may be useful for logging.
# can be done through inspect.

# questions about the model.

# stuff to manipulate existing experiments folders.
# working in terms of relative paths is crucial for this functionality.
# for example, generating scripts for running the code in a certain way

# another possibility would be to 

# add functionality for relative directory manipulation.

# TODO: in some cases, it is possible to output things to terminal too.
# this means adding some tee to the model.

# to get the full encapsulation, I will need to do something else.
# this is going to be interesting.

# decrease by when and how much. question about the different part of the 
# model. this is an interesting question.

# add dates to the experiments, or at least make it easy to add dates.
# this can be done in main_draft.

# stuff for handling more complicated data, such as data bases and other type 
# of data.

# it may be worth to introduce information about some computational resources.

# some thing for, if a certain file exists, ignore, or somethigng like that.

# TODO: add the download of only files of a certain type.'

# add ignore lists to file transfer and what not. 
# it is better to compress before transfering, probably.

# look into rsync. this is something that might be interesting.

# TODO: add the information about the model 

# using stuff in slurm, this is going to be important.

# there is definitely the possibility of doing this more seriously for using 
# slurm commands. the question is how far can I take this.

# tools for taking a set of dictionaries, and sorted them according to 
# some subset of the keys. this can be done easily. 
# manipulating lists of dictionaries should be easy.

# TODO: all the sorting elements must exist.

# it would be interesting to have some form of interrupts that allow a 
# process to stop, or to call a function with some time limit, but 
# to terminate it if necessary.

# the important thing is that there are multiple models that can be used.
# for example, it can be possible to try different models and check that it is 
# going to work in certainn way.

# make  it easy to run some of the search algorithms in parallel or something 
# like that. what can be done? it is important to set up a database or 
# have a central thread that takes care of this. interprocess communication
# to do this. very simple code.

# there is also stuff to do the analysis of the model.

# handling different data types very easily. this is going to be something 
# interesting.

# the main components are data and model. add some functionality do 
# deal wtih both of them.

# abide by some interface where it possible to pass some logging information
# using some consistent interface to the logging functionality, and just pass
# that around. 
# that would be important for getting things to work properly

## TODO: perhaps I can add information about how to make this work. 
# 

# important for visualization of mistakes of a model. how to inspect the mistakes?
# perhaps write a json, or just write some form of function that takes
# some performance metric, and write all examples and predictions for 
# visualization.

# add a condition to test if running on the server or not. this 

# check if running on the server or not.

# maybe the best way of doing the import of matplotlib is to do it, is to do a 
# try and raise.

# TODO: tools for checking that all the experiments are complete, and perhaps 
# for removing the ones that are not complete.
# there is somemore additional processing of results that would be interesting.

# add more functionality to streamline profiling. it should be easy to extract
# relevant information. look at the pstats objects.

# clearly, the question of running multiple jobs on lithium is relevant.
# look into this. use a few gpu machines.

# add scripts for dealing with slurm and the other one.
# interface by having simple interfaces that generate the scripts.
# such it is simply doing srun on the script.

# TiKz figures. easy to manipulate. for simple cases. e.g., block diagrams 
# or trees. getting some consistency on the line width would be important.
# how can I do a more consistent job than the other one.

# do some more of this for some simple bash scripts.

# TODO: have some way of representing attention over the program given 
# the error message and perhaps some more information. this is going to be 
# important. the other question is instrumentation. 
# if I have a small dataset, keeping the field names with it is not so bad.

# also, if the dataset is a sequence of time events, that is kind of trickier
# to collect. or is it? maybe not.

# datasets collected as soon as possible, or collected incrementally.
# define fields and keep adding to them. serialization through json.
# loading through json.
# or perhaps csv. depends on the creation of a new file.

# depends on the 

# take a script and make it some other type of script. example, 
# from running locally to running on the server.

# do the experiments in both the cluster and here.

# have some utility scripts that can be called from the console.
# also, perhaps, some way of easily creating some files and iterating over files.
# it is important to work easily with them

# module load singularity 
#   exec centos7.img cat /singularity

# there are only a few that 

# it is easy to preappend with what we have done.

# something for wrapping the experiments, I guess.

# def make_slurm_with_tf(exp_folderpath):
#     "module load singularity"
#     "singularity run %s" % exp_folderpath

# problems defining some of these scripts. version is too 

# there is probably some way of redirecting output.

# to run scripts that input, it is 

# generating easily the scripts for running on the machine. do not worry about 
# having too much stuff now.

# TODO: I will need some form of managing experiments across projects.

# handling the experiments some how. this would be possible easily.

# easy to send things to server. 
# should only take a few minutes.

# make a script to install some of these things easily.
# this will make setting up easier.

# sort of incremental contruction of the dictionary which is constant
# pushed to disk
# write to pickle file.

# add a command to call bash commands.
# this is going to make it easier getting access to some of information 
# from the command line.

# write a lot of reentrant code. make every cycle count.

# set up the repos; run more experiments for beam_learn; do more stuff.

# do different commands for running in different servers there.
# it should be possible to run a command directly after starting a session there.

# NOTE: remember that it is nice to run this thing from command line. 
# there is still the question about creating these things through the command 
# line. the question is how to manage multiple functionalities.

# NOTE: for having something about key identification for not having a 
# password prompt for connecting to the server.

# TODO: add some functions to manipulate csv files to generate tables about 
# quantities that I care about.

# add the results.json file.
# file. write json file with just the fields that matter.

# TODO: handle better the continuum of steps that can be taken. 
# this means 

# TODO: tests for whole experiments in the sense that is mostly about what do we 
# expect to happen and why is it not happening. it is just going to help going 
# through the experiments the first time.

# TODO: the project should be easy to manage. it should be possible to 
# read the data from disk, do something and then, put it back on disk.

# map a file line by line somehow.

# TODO: have mailing functionalities upon termination of the job.
# this sounds something reasonable if I want to check the results soon.

# NOTE: for experiments, it is important to maintain consistency between 
# the remote version and the local version.

# TODO: have a way of syncing folders

# possible to have character escaping there.

# TODO: there is the question of on which compute node was it ran. 
# that can be done through sacred, if I really want to be 

# TODO: functions for file processing. like taking a file with a huge number of 
# functions and just keeping those that I care about.
# this is clearly processing the file. and then, keep only the imports 
# used at the top. NOTE: that there is some form of dependency because some 
# auxiliary functions may use some other auxiliary functions, 
# there is some recursion that I need to go about.

# TODO: make it easy to run callibration experiments to get a sense of how 
# much an epoch is going to take. also, I think that it would be interesting 
# to have some form to actually talk about epochs and performance. that would 
# make it easier.

# NOTE: have a simple way of running any command of the toolbox on the command
# line. I think that this is only possible for commands with string arguments 
# or I have to anottate the commands with tags.
# --int --str --float 
# --bool is currently unsupported

# these are going to especially useful for file manipulation, and for file 
# file creation.

# TODO: I think that it would also be interesting to have a way of running a 
# folder somehow. 

# extracting the requirements from running some command

# manipulation of the model.

# TODO: run functions from the command line. it can also list all functions.
# not all are callable.

# TODO: check how to fix the randomness for things such as tensorflow 
# and other problems.

# comparison of different experiments is going to be important, I think that 
# this is going to be mostly in the analysis code for plot generation and 
# other things.

# NOTE: preprocessing is going to be quite interestin, but I need 
# to be careful about how do I detect that something is a function call.

# TODO: there is probably some more interesting prints, but I do not 
# know exactly what do I want.

# to match a function call, basically, the name of the function followed by 
# a parens.
# can first parse the definitions and do that. Note that this is important 
# for generating the toolbox in a way that I'm happy sharing.
# also perhaps for reducing the size of file.

# TODO: something to remove all comments of a file.

# TODO: something to call from the command line to strip a file of comments 
# somehow. 
# in the case of python, perhpas move all imports to the top.
# NOTE: have a simple way of doing this processing in a file.
# NOTE: it is good to have something that allows me to go quickly and seamlessly 
# between the terminal and the python interpreter.

# NOTE: something to consider would be

# another interesting aspect is having something to remove all comments, or 
# just comments outside functions.

# the file manipulation part can be used effectively outside the shell.
# one high level goal is to have most of the operations be done outside the 
# function.

# TODO: there is stuff on data loading and data buffering that would be 
# interesting to consider. I don't know what are the main aspects that make 
# it possible to do it? is it just multithreaded code? check this out.
# have stuff that allows you to easily set up a model in different languages.

# TODO: perhaps also make it easy to submit jobs from the command line, 
# even it requires creating other things.

# TODO: develop tools for data collection. this means that I can register 
# information and stuff like that. 
# event data and stuff like. register certain things. simple data types.

# also, have th capacity of going seamlessly between disk and memory.

# for example, it should be possible to do everything with the experiments
# regardless of how much time passed between they have been created.
# failing to do this, means that you cannot reuse the information there. 

# TODO: provide a clock object that allows you do get time and date.
# and both in 

# TODO: study epoch and convergence curves and see what is the distribution
# of performance for the models. 

# TODO: it should be as simple to keep a dataset, and it is to keep code.
# data should be easy to generate and maintain. 

# TODO: data should be easy to change, and keep in place. data is effectively 
# the most important thing in ML.

# TODO: easy to crawl data and tools. how to?

# these in memory datasets are quite powerful, as we could read them and 
# write them to memory again. the append mode is better, but the
# question about keeping data is quite important. perhaps git works alright
# for simple things, but it is suboptimal. log data or event data.

# NOTE: the definition of data is essentially tied to a reader and writer, 
# and some form of registering stuff.

# training for longer or having longer patienties seems to make a difference 
# in running the models. 

# TODO: have some typical configs, like calibration, fast, noop, default.
# calibration is an experiment, rather than a config, note that a run is not 
# independent of the machine it was ran in.

# TODO: it should be possible to very easily extract a few quantities from 
# log files, do this. lines that do not match the pattern can be ignored.
# notion of a sequence of objects of the same type that can be processed easily.
# extracting information from files is very important.

# TODO: extracting information 

# TODO: perhaps I can do different tracking of the learning rates, or of 
# what constitutes an improvement. I think that asking for the best is a little
# strict. I think that it is better to keep a running average.
# if overall is improving, I think that it is better. maybe be less aggresive.

# TODO: it should be easy to serialize models and load them again. 
# it should be easy to run models on default data.

# typical data:
# - sequences
# - images
# - paired data
# - sequences of floats.

# it would be interesting to define these data types and work directly with them
# it would make it easier to do certain things.

# main preprocess, main train, main evaluate, main analyze 
# main all or something.
# should be easy to run all these things  
# what is the advantage of doing this?

# should be easy to run a sequence of these one after the other.
# it should be easy to keep a model around for future inspection.

# it should be easy to run a model on some data, just from the command line.
# I think that one the the main things is 

# extract data for graphs. 

# extract data from tree structures, like the simple deeparchitect with single 
# input, single output

# ability to interface with simpler code in C or something like that.
# that would be nice to do.

# have a research toolbox in C. for simple operations and simple data 
# structures. stuff like that.

# NOTE: perhaps the creation of certain code in C can be done through 
# Python, or at least some aspects can be generated, like functions, 
# argument types, and perhaps memory manipulation. 
# other interfaces that make sense can also be done there. I don't think 
# it is very easy to generate code in python for C. 

# TODO: interface for a model and a model trainer.

# TODO: have easy functionality for conditional running, e.g., if a file exists
# or not.

# if the operations are only a few, maybe the generation of the code can be 
# done in Python, otherwise, it seems hard to do. For example, if the 
# operations are mostly about doing some arithmetic operations, then it 
# should be possible to do them in Python, and define the data structures 
# accordingly.

# everything should be possible to do programatically. 
# the question is what is the benefit of doing this way rather than just doing 
# it. I think that 

# TODO: have a function to clear a latex file from comments. useful for submissions.
# careful about licenses. maybe be necessary to keep some licenses in some files.

# interface for data.

# interface for different tasks for putting together a model really fast.
# some general formats. 

# TODO: doing fine-tuning should be easy.

# TODO: have a sequence of patiences or something like that, or only tune a 
# few hyperparameters. 
# TODO: easy interfacing with DeepArchitect for setting search spaces.
# easy to train a few models based on some input representation.

# interface to the data should be easy.

# simple featurizer for sequences. 

# TODO: what about sharing the weights.

# TODO: always need to think about what is it going to be the representation for 
# the base models. doing this right is important to save computation. it si 
# better to keep the data in that format.

# TODO: models from graph data. 
# the question is the minimum information that I can use about the model.
# interface for setting up an easy LSTM model and easy Convolutional Net.
# these models should be super easy to use and load.
# the definitions and simple. the training should also be simple.

# I think that often the most important thing is just training for long.

# TODO: for preprocessing code files, have a way of sorting the function 
# definition by name or by number of args or whatever. number of times it 
# appear somewhere.

# it should be easy to have a folder and do something with it. for example,
# get the data at the end of the experiment. 

# tensorflow model manipulation. 
# pytorch model manipulation.

# make sure that you can address pytorch pitfalls.

# TODO: the server should feel like an extension of the local machine in 
# that it is simple to push things to the server and pull them from the server.
# it is easy to ask the server about how it is going.

# TODO: have an easy way of setting up email reminders when experiments 
# complete. do periodic emails with the state, and how are things going.

# TODO: minimum number of reductions before next reduction. this is 
# for the step size patience and stuff like that. it should help running things.

# easy error analysis, like  what are some of the examples that failed.

# the inspect module is nice for code manipulation.

# it should be easy to take a folder and run it on the cluster with some 
# configuration, then it should be easy to get the results back, maybe with
# locking or not.

# TODO: train something that knows when to reduce the step size of the models 
# based on the current step size, and the amount of variation observed.
# these should ideally be trained for a single model.
# there is also a question about normalization and stuff like that.

# TODO: add some preprocessors for different types of numbers. basically, 
# things that can be computed after you register all the training data.

# experiment sequencing is important, but I don't know exactly how to do it.
# it may be inside the cfg for that experiment, or it may be outside the experiments.
# it may be easier because it shows the dependencies, but I think that it 
# becomes too fragmented. if I have functions to go over all folders,
# I think that it is doable, but it is still trouble some.

# TODO: what can we do to tune the step size parameter?
# TODO: what to do when there? it is kind of like some form of optimization.

# TODO: how to deal with things where observations are sequences? I think that 
# it can be consumed with an LSTM. 

# TODO: implementation of simple hierarchical LSTM, maybe a single level or 
# something like that.

# I want it to be really easy to take something and just try something 
# TODO: perhaps add tools to make the setup easy too.

# run some of those models.

# add some additional conditions that help to detect if the model is training 
# or not.

# add support to some other plotting libraries, like d3 and other relevant 
# ones. 
# there is also graphviz that may be interesting for certain aspects of the code.

# TODO: add some default model optimization that uses some reasonable decrease 
# schedule, or tries a reasonable set of hyperparameters for the typical things.
# it is going to be important to check this.

# NOTE: if running things directly from the command line, I think that it is 
# going to be easy to run this code, even if doing a remote call. 
# perhaps printing to the command line is the way to go, easy 

# TODO: it should be easy to set sweeps with respect to some parameters.
# I think that it is, actually.

# TODO: using inspect, it is possible to run a function or at least to 
# get the arguments for a function.

# easy to store sequences of sequences? json files for that.

# TODO: add in the run script the option to only run if it does not 
# exist already. I think that results.json is a good way of going about it, 
# but probably not quite. the question is where should we test if something should
# be done or not? that is, to run or not.

# running those that have not been ran yet. that sounds good.
# also, maybe run guarded experiments for these. I don't think that it should 
# take too much time.

# what is a good description of a featurizer. I think that the definition of 
# the data is relatively stable, but featurizers are malleable, and depend 
# on design choices. the rest is mostly independent from design choices.

# binary search like featurization. think about featurization for 
# directory like structures.
# this is binarization. check how they do it in scikit-learn.

# manipulating the code should be easy because we can solve the problem
# by 

# download certain files periodically or when it terminates. this allows to 
# give insight into what is happening locally.
# there is syncing functionality 

# TODO: make it easy to use the sync command.

# managing jobs after they are running in the server.

# TODO: write a file with information about the system that did the run.
# gathering information about the system is important

# TODO: have a way of knowing which jobs are running on the server (i.e., which 
# are mine.) or keep this information locally. this can be done asyncronously.
# keep this information somewhere. 

# TODO: and perhaps kill some of the jobs, or have some other file 
# that say ongoing. define what is reasonable to keep there.

# NOTE: make a single experiment. it does not make sense to have multiple 
# experiments with the same configuration. easier to keep track this way
# up to three word descriptive names.

# 09-06-2017_....
# labelling experiments like this can work, I think.

# TODO:
# Calendar
# get day, get hour, get minute, get year, get month, for some, there is the 
# possibility of asking for 
# get_now(...): 
# choose whether you want as a string or number.

# the question is when the experiment is created. there should exist a single
# experiment. run_only_if_notexists. there is the question of running the code.
# this is easy, check 
# there is some. has an option to override. 
# test if a given file exists, if yes, then it does not run the experiment.
# NOTE: making everything index to the place where it is is nice.

# NOTE: to control randomness, it is necessary to set it on every call to the 
# function. that is non-trivial. look at what sacred does.

# TODO: have a simple way of syncing the folder for running experiments 
# somewhere. it should be really easy to sync the experiments on those folders.

# TODO: perhaps sync remote folder through the command line.
# it is sort of a link between two folders.

# I don't think that the current one is correct.

# find a nice way of keeping track of the experiments having a certain 
# configuration. it helps thinking about what I am doing.

# TODO: use cases about making things work. 

# sync two folders or multiple folders

# creation of all the models 

# do something about the script to get it to work 
# just do a loop, and fix the number of tasks and keep things there.

# only keep the arguments that matter in writing this thing to disk.

# it is hard to draw conclusions from random experiments. 
# typical parameters to vary are 
# computational capacity.
# amount of data
# time budget.

# TODO: add stuff to use hyperparameters optimization really easily
# stick with the str, int, float, bool types.
# NOTE that for strings, I probably have to give it a set, it is a discrete
# set of models.

# there is a question about the run script, and what can be done
# register a daemon, do something periodically, for example, reads the folder 
# there are gets the files.

# there is stuff that we can do with rsync. there is stuff that I can ignore.

# TODO: there is stuff that is part of the analysis, like the 

# NOTE: in syncing folders, it is important to keep the most recent ones
# mostly from the server from.
 
# should be easy to run multiple commands remotely.

# NOTE: it is  easy to define a function locally and then update it to the
# server.

# testing code for interacting with the server is nice because I can stop if 
# it fails to reach it, so I can iterate fast.

# NOTE: it mostly makes sense to get stuff from the server to the 
# local host.

# basically there is stuff on the upload link, and there is stuff on the 
# download link. on the upload link, I just want to sync some of the 
# experiments.

# NOTE: I think that for the most part, I will be syncing folders in the 
# same server. they may be different across experiments, but that is not 
# necesarily true.

# TODO: think about introducing the configurations in a way that they are 
# related.

# how to keep different experiments. also, the fact that the code sort of 
# changes with the time the model is run is also something important. 
# the experiments folder can be done 

# the only way of getting this for sure is to have a copy of the code at the 
# moment it is done.

# the expectations about running about running and what can I do regarding 
# different things that make a difference. for example, what is the interest 
# in having replication in the configs? maybe there is no interest, but there 
# is a question of entry point, I guess. not even that, the entry point is 
# the same in both cases. It would be better to have just one or the other
# I think that the config would be better.

# why would you change an experiment configuration. only if it was due to 
# bug. what if the code changes, well, you have to change the other experiments
# according to make sure that you can still run the exact same code, or maybe 
# not. 

# it does not make much sense, because they may become very outdated.

# the fact that I have to run a command on a server makes it that I have to 
# generate a file many times.

# TODO: add to the toolbox functionality to tune the step size.

### older stuff that has been moved.

# TODO: difference between set and define.
# set if only exists.
# define only if it does not exist.
# both should have concrete ways of doing things.

# TODO: perhaps it is possible to run the model in such a way that makes 
# it easy to wait for the end of the process, and then just gets the results.
# for example, it puts it on the server, sends the command, and waits for 
# it to terminate. gets the data everytime.

# it has to connect for this...

# TODO: the question is which are the cpus that are free. 
# NOTE: it is a good idea to think about one ssh call as being one remote 
# function call.

## TODO: get to the point were I can just push something to the server 
# and it would work.

# TODO: perhaps it is possible to get things to stop if one of the 
# them is not up.
# I'm sure that this is going to fail multiple times.

# these tree wise dictionaries

# this is nice for dictionary exploration

# TODO: it is probably a good idea to keep information about the model.

# ideally, you just want to run things once.
# also, perhaps, I would like to just point at the folder and have it work.

# TODO: develop functions to look at the most recent experiments, or the files
# that were change most recently.

# also stuff to look into the toolbox.
# the plug and play to make the experience 

# check multiples of two, powers of two, different powers too.

# stuff for adapting the step size. that is nice.

# remember that all these are ran locally.

# the commands are only pushed to the server for execution.

# easy to do subparsers and stuff, as we can always do something to
# put the subparsers in the same state are the others.
# can use these effectively.

# TODO: add something to run periodically. for example, to query the server
# or to sync folders. this would be useful.

# TODO: split these commands in parts that I can use individually.
# for example, I think that it is a good idea to 
# split into smaller files. 
# 3-7; not too many though.

# TODO: it is possible to have something that sync folders
# between two remote hosts, but it would require using the 
# current computer for that.

# TODO: get it to a point where I can call everything from the command line.

# to extract what I need to get, I only need to to use the model
# very simply.

# TODO: understand better what is the pseudo tty about.

# add error checking 

# TODO: check output or make sure that things are working the correct way.
# it may be preferable to do things through ssh sometimes, rather than 
# use some of the subprocess with ssh.

# NOTE: it is important to be on the lookout for some problems with the 
# current form of the model. it may happen that things are not properly 
# between brackets. this part is important

# there is questions about being interactive querying what there is still 
# to do there.

# there is only part of the model that it is no up. this is better.

# there is more stuff that can go into the toolbox, but for now, I think that 
# this is sufficient to do what I want to do.

# add stuff for error analysis.

# TODO: functionality for periodically running some function.

# TODO: more functionality to deal with the experiment folders.

# for reading and writiing csv files, it is interesting to consider the existing 
# csv functionality in python

# look at interfacing nicely with pandas for some dataframe preprocessing.  

# NOTE: neat tools for packing and unpacking are needed. this is necessary 
# to handle this information easily.

# TODO: mapping the experiment folder can perhaps be done differently as this is 
# not very interesting. 

# dealing with multiple dictionaries without merging them.

# going over something and creating a list out of it through function 
# calls, I think that is the best way of going.

# TODO: a recursive iterator.
# recursive map.

# NOTE: a list is like a nested dictionary with indices, so it 
# should work the same way.

# NOTE: for flatten and stuff like that, I can add some extra parts to the model
# that should work nicely, for example, whether it is a list of lists 
# or not. that is nicer.

# there are also iterators, can be done directly. this is valid, like 
# [][][][][]; returns a tuple of that form. (s, s, ... )
# this should work nicely.

# question about iterator and map, 
# I think that based on the iterator, I can do the map, but it is probably 
# a bit inefficient. I think that 

# the important thing is copying the structure.

# NOTE: some of these recursive functions can be done with a recursive map.

# most support for dictionaries and list and nested mixtures of both.
# although, I think that dictionaries are more useful.

# <IMPORTANT> TODO: it would be nice to register a set of function to validate that 
# the argument that was provided is valid. this can be done through 
# support of the interface. improve command line interface.

# TODO: make it easier to transfer a whole experiment folder to the server and 
# execute it right way.

# it seems premature to try a lot of different optimization parameters until
# you have seen that something works.

# develop tools for model inspection.

# TODO: some of the patterns about running somewhere and then 

# TODO: some easy interface to regular expressions.

# TODO: stuff to inspect the examples and look at the ones that have the most 
# mistakes. this easily done by providing a function

# TODO: there are problems about making this work.

# something can be done through environment variables, although I don't like 
# it very much.

# there is stuff that needs interfacing with running experiments. that is the 
# main use of this part of this toolbox.

# think about returning the node and the job id, such that I can kill those 
# job easily in case of a mistake.

# TODO: add stuff for coupled iteration. this is hard to do currently.
# think about the structure code.

# TODO: work on featurizers. this one should be simple to put together.

# NOTE: it is a question of looking at the number of mistakes of a 
# model and see how they are split between types of examples.

# or generate confusion matrices easily, perhaps with bold stuff for the largest
# off-diagonal entries.




# Working with the configs, I think that that is interesting.
# the folders do not matter so much, but I think that it is possible 
# do a mirror of a list [i:]

# TODO: can use ordered dict to write things to 

# this would work nicely, actually. add this functionality.
# no guarantees about the subtypes that are returned.
# this could be problematic, but I think that I can manage.

# TODO: the point about the checkpoints is that I can always analyse the 
# results whenever I want.

# <IMPORTANT> TODO: add more iterators, because that would be nice.

# TODO: add logging functionality that allows to inspect the decisions of the 
# model, for example, the scores that were produced or to re

# TODO: implement.

# generating two line tables. some filtering mechanisms along with 

# TODO: add ML models unit tests.
# like running for longer should improve performance.
# performance should be within some part of the other.
# do it in terms of the problems.

# <IMPORTANT> TODO: logging of different quantities for debugging.
# greatly reduce the problem.
# copy the repository somewhere, for example, the debug experiments.

# do it directly on main.

# <IMPORTANT> TODO: simple interface with scikit-learn

# unit tests are between different occurrences, or between different time steps.

# TODO: these can be improved, this is also information that can be added 
# to the dictionary without much effort.
import platform

def node_information():
    return platform.node()


# TODO: function to get a minimal description of the machine in which 
# the current model is running on. 

### TODO: check the overfit test, meaning that I can 
# 

# loss debugging. they should go to zero.
# easy way of incorporating this.

# both at the same time.

# TODO: dumb data for debugging. this is important to check that the model is working correctly.

# question about the debbuging. should be a sequence of 
# objects with some descriptive string. the object should probably also
# generate a string to print to a file.
# this makes sense and I think that it is possible.

# TODO: a logging object.
# can have a lot of state.

# NOTE: how should logging look like? it should tell the path. of the model. where is being called, and the 
# ids and stuff. 
# inspection module.

# conditions that should be verified for correctness. 
# conditions that should be verified. 

# how to inspect the results. perhaps have a good spreadsheet?

# that would be nice.

# TODO: for example, if I'm unsure about the correctness of some variable
# it would be nice to register its computation. how to do that.
# I would need to litter the code with it.

# also, can describe certain things that allow to inspect the log.
# it could a dynamic log file. have certain namespaces.

# TODO: passing a dictionary around is  a good way of registering informaiton 
# that you care about. this is not done very explicitly. using just a dictionary 
# is not a good way. perhaps reflection about the calling function would be 
# a good thing, and some ordering on what elements were called and why.

# TODO: have a function to append to a dictionary of lists.

# this registering is important though. how can you do this automatically
# that would be some form of what

# using nested dictionarie s 

# with the parameters that determine computation to be very small such that it
# can be ran very fast.

# runs directly from config files and result files? 
# what would one of these look like for DeepArchitect? 
# very small evaluation time, simple search space.

# TODO: add gradient checking functionality.

# TODO: add some easy way of adding tests to the experiment folders. like 
# something as to be true for all experiments.
# also something like, experiments satisfying some property, should 
# also satisfy some other property.

# NOTE: bridges is slurm managed. I assume it is only slighly different.


# do soemthing to easily register conditions to test that the experiments 
# should satisfy.

# add a non place holder for table generation.
# there must exist at most one element for the creation of this table.
# the other stuff should be common. 

# add a function to say which one should be the empty cell placeholder in the 
# table.

# TODO: stuff for error analysis. what are the main mistakes that the model 
# is doing. there should exist a simple way of filtering examples by the number
# of mistakes.

# curious how debugging of machine learning systems work. based on performance.
# differential testing.

# TODO: it should be easy to specify some composite patience clock.

# TODO: generate powers of two, powers of 10, 

# have default values for the hyperparameters that you are looking at.
# for example, for step sizes, there should exist a default search range.\

# TODO: question about debugging models

# TODO: do the featurizer and add a few common featurizers for cross product
# and bucketing and stuff.

# think about how to get the featurizer, that can be done either through 
# it's creation or it is assumed that the required information is passed as 
# argument.

# this is a simple map followed by a sort or simply sorting by the number of 
# mistakes or something like that.
# perhaps something more explicit about error analysis.

# generation of synthetic data.

# TODO: think about the implementation of some stateful elements. 
# I think that the implementation of beam search in our framework is something 
# worthy.

# simple binarization, 

# TODO: check utils to train models online. there may exist stuff that 
# can be easily used.

# TODO: simple interface with DeepArchitect.

# TODO: interface with Pandas to get feature types of something like that 
# I think that nonetheless, there is still information that needs to 
# be kept.

# one simple way of getting these models is by doing a Kaggle competition, 
# and winning it.

# TODO: stuff to manage state. for example, feature updates when something 
# changes, but otherwise, let it be.

# this can be done in the form of something that is ran when a condition is 
# satisfied, but otherwise does nothing.
# the condition may be having changes to the state and calling it
# this is lazy computation.

# how to switch once it is finished.

# tests to make the model fail very rapidly if there are bugs.
# functions to analyse those results very rapididly.

# TODO: stuff for checkpointing, whatever that means.

# add functionality to send email once stops or finishes. 
# this helps keep track of stuff.

# TODO: some easy interfacing with Pandas would be nice.
# for example to create some dictionaries and stuff like that.

# very easy to work on some of these problems, for example, 
# by having some patterns that we can use. this is interesting in terms of 
# model composition, what are typical model composition strategies?
# for example, for models with embeddings.

# TODO: functionality to work with ordered products.

# NOTE: perhaps it is possible to keep scripts out of the main library folder,
# by making a scripts folder. check if this is interesting or not.
# still, the entry point for running this code would be the main folder.
# of course, this can be changed, but perhaps it does not matter too much.

# NOTE: what about stateful feature extraction. that seems a bit more 
# tricky.

# there is the modelling aspect of the problem. what can be done.

# TODO: axis manipulation and transposition.

# TODO: online changes to configurations of the experiment. 
# this would require loading all the files, and subsituting them by some
# alteration. this is kind of like a map over the experiments folder. 
# it does not have to return anything, but it can really, I think.
# returning the prefix to the folder is the right way of doing things.

# make it easy to do the overfit tests. 
# it is a matter of passing a lot of fit functions. this can be 
# done online or by sampling a bunch of models from some cross product.

# try a linear model for debugging. try multiple.
# differential testing.

# sorting and stuff like that is a good way of 

# function to convert to ordered dict.


# NOTE: some of these aspects can be better done using the structure 
# information that we have introduced before.

# TODO: add functionality to make it easy to load models and look at it
# an analyse them.

# TODO: error inspection is something that is important to understand

# some subset of the indices can be ortho while other can be prod.

# TODO: in the config generation, it needs to be done independently for 
# each of the models. for example, the same variables may be there
# but the way they are group is different.

# TODO: for the copy update, it is not a matter of just copying
# I can also change some grouping around. for example, by grouping 
# some previously ungrouped variables.
# it is possible by rearranging the groups.
# the new grouping overrides the previous arrangement. 
# this may be tricky in cases, where the previous ones where tied.
# perhaps it is better to restrict the functionality at first.

# the simple cases are very doable, the other ones get a little more tricky.

# composite parameters that get a list of numbers are interesting, but have
# not been used much yet. it is interesting to see 
# how they are managed by the model.

# managing subparsers, like what is necessary or what is not is important
# for example, certain options are only used for certain configurations.

# TODO: add the change learning rate to the library for pytorch.

# TODO: I have to make sure that the patience counters work the way I expect 
# them to when they are reset, like what is the initial value in that case.
# for example, this makes sense in the case where what is the initial 
# number that we have to improve upon.
# if None is passed, it has to improve from the best possible.

# the different behaviors for the optimizers are nice and should work nicely 
# in practice.
# for example, the training model that we are looking at would have a 
# update to the meta model within each cost function

# for the schedules, it is not just the prev value, it is if it improves on the 
# the prev value or not.

# TODO: functionality to rearrange dictioaries, for example, by reshuffling things
# this is actually quite tricky.

# simply having different models under different parts of what we care about.

# make better use of dictionaries for functions with a variable number of 
# arguments.

# TODO: have an easy way of doing a sequence of transformations to some 
# objects. this is actually quite simple. can just just


# TODO: do some form of applying a function over some sequence, 
# while having some of the arguments fixed to some values. 
# the function needs to have all the other arguments fixed.

# TODO: also make it simple to apply a sequence of transformations 
# to the elements of a string.

# functional stuff in python to have more of the data working.
# for exmaple, so simple functions that take the data 


# think about unpacking a tuple with a single element. how is this different.

# TODO: dict updates that returns the same dictionary, such that they 
# can be sequenced easily.

# TODO: easy to build something that needs to be ran in case some condition 
# is true.

# easy to run things conditionally, like only runs a certain function 
# if some condition is true, otherwise returns the object unchanged.

# TODO: a trailling call is going to be difficult, as it is not an object in 
# general, nesting things, makes it less clear. otherwise, it can 
# pass as a sequence of things to execute. looks better.
# conditions.

# NOTE: a bunch of featurizers that take an object and compute something based 
# on that object. featurizers based on strings an 

# sequence of transformations.
# use better the vertical space.

# keep stuff like job and node id in the config.

# the toolbox about the counters needs to be more sophisticated.

# TODO: add variable time backoff rates.

# TODO: code to run a function whenever some condition is verified.
# TODO: code to handle the rates and to change the schedules whenever some 
# condition is verified.

# TODO: stuff with glob looks nice.

# TODO: an interesting thing is to have an LSTM compute an embedding for the 
# unknown words. I think that this is a good compromise between 
# efficiency and generalization.

# NOTE: this is going to be difficult.

# NOTE: some auxiliary functions that allows to easily create a set of experiments 
# there are also questions.

# stuff to append directly to the registered configs.

# partial application is sufficient to get things to work nicely.

# it is a matter of how things can be done.


## NOTE: something like this such that it just loads the weights and then 
# it is ready to apply it to data.
# class PretrainedModel:

#     def predict(self, ):
#         pass

# TODO: it is important to generate a csv with a few things and be able to look 
# at it. 
# tables are easy to look for sweeps, but it is harder to understand 
# other aspects.

# it is harder to read, but I think that with a sorting function becomes
# easier.

# each config may have a description attached.

# NOTE: some of the stuff from the plots can come out of current version of
# of the plots.

# NOTE: possibility of a adding a few common experiment structures.

# easy way of doing subexperiments, but with prefixes and stuff.

# NOTE: I think that it is probably important

# the run script for a group of experiments is going to be slighly different
# what can be done there. it would be inconvenient for large numbers to send 
# it one by one. I guess it it possible to easily do it, so not too much of a 
# problem.

# TODO: ortho increments to a current model that may yield improvements.
# or just better tools to decide on what model to check.

# I think that it is going to be important to manage the standard ways of 
# finding good hyperparameters.


### NOTE: this function can be adapter to read stuff from a column wise format 
### file, perhaps with some specification of how things should go about dealing
# with conversions of certain colunns.
# def load_data(fpath, word_cidx, tag_cidxs, lowercase):
#     with open(fpath, 'r') as f:
#         sents = []
#         tags = []
#         s = []
#         t = []        
#         for line in f:
#             line = line.strip()
#             if line == '' and len(s) > 0:
#                 sents.append(s)
#                 tags.append(t)
#                 s = []
#                 t = []
#             else:
#                 fields = line.split()
#                 if lowercase:
#                     s.append(fields[word_cidx].lower())
#                 else:
#                     s.append(fields[word_cidx])
#                 t.append( [ fields[i] for i in tag_cidxs] )
            
#     return (sents, tags)

### TODO: I think that this is a more appropriate part of the model.
# it makes more sense to keep things like this.

# # # load some of the columns in a text file.
# def load_data(fpath, idxs):
#     assert len( idxs ) > 0

#     with open(fpath, 'r') as f:
#         rs = [ [] for _ in idxs ] 

#         rs_i = [ [] for _ in idxs ]        
#         for line in f:
#             line = line.strip()
#             if line == '' and len( rs_i[0] ) > 0:

#                 # append each of the senteces to the ones already present.        
#                 for r, ri in zip(rs, rs_i):
#                     r.append(ri)

#                 rs_i = [ [] for _ in idxs ]        

#             else:
#                 fields = line.split()

#                 for i, idx in enumerate(idxs):
#                     rs_i[i].append( fields[idx] )
#     return rs

# this can be convenient to read csv files.
# reading a csv file is easy if you do not want to convert things to other 
# format.


# manipulation of the different parts of the model is important because I'm 
# not sure about how to go about it.

# TODO: handle cases with variable output depending on the arguments that are 
# passed. returning a dictionary seems like a good compromise.

# TODO: stuff to inspect the mistakes that wer

# TODO: add function to check everybody's usage on matrix. same thing for 
# lithium and matrix.

# TODO: functions to inspect the mistakes made by the model.

# NOTE: some of these consistency checks can be done automatically.

# TODO: something to randomize the training data, and to keep some set of sentences
# aligned.
# TODO: some tests like checking agreement of dimensions.
# or for paired iteration: that is pretty much just izip or something like that.

# TODO: another test that may be worth validating is that the model, 
# may have some property that must be satisfied between pairs of the models.
# using stuff in terms of dictionaries is nice.

# TODO: splitting stuff is tricky. 
# that needs to be done easily 
# st

# TODO: better way of maintaining invariants in Python.
# how can that be done.

# TODO: differential testing, given the same prediction files, are there mistakes 
# that one model makes that the other does not.
# having a function to check this is interesting.


# stuff to go back and train on all data. check this.

# NOTE: it is important to ammortize effort. is it possible to write aux functions 
# for this.

### some useful assert checks.
# NOTE: this is recursive, and should be applied only to sequences 
# of examples.

# TOOD: something that can be computed per length.

### TODO: another important thing is managing the experiments.
# this means that

# TODO: perhaps important to keep a few torch models 
# and stuff.

# TODO: the tensorflow models can be kept in a different file.

# TODO: add code to deal with the exploration of the results.
# TODO: also some tools for manipulation with LSTMs.

# TODO: stuff to build essembles easily.

# TODO: add stuff for initial step size tuning.

# TODO: add functionality to run DeepArchitect on the matrix server.


# TODO: check on 

# TODO: stuff on synthetic data.


### Categories:
# - utils
# - io
# - experiments
# - plotting
# - compute
# - analysis (this is kind like plotting)
# - system
# - optimization
# - preprocessing
# - data
# - tensorflow
# - pytorch
# - (other stuff for )
# - what about the featurizer.
# - asserts and stuff.

# TODO: find a good way of keeping these categories in a way that 
# makes sense.


# TODO: add funcitons that do high level operations.

# TODO: finally finish using 

# TODO: add function doing cross validation, that is then ran on the 
# full dataset.

# TODO: batchification

# TODO: stuff for cross validation . it has to be more general than the 
# stuff that was done in scikit-learn.

# TODO: best of three runs.

# TODO: integration with other functionality that is simple to run.

# NOTE: the worskpace gets very messy.
# the problem is dealing with the scheduling of all these parameters.
# it may be difficult.
# can have multiple schedules that require some informaiotn.
# perhaps publishing subscribing, but the only purpose is really just free the search 
# space.

# managing validation.

# are there things that are common to all models and all experiments.

# TODO: stuff to apply repeatedly the same function.

# handling multiple datasets is interesting. what can be done there?

# how to manage main files and options more effectively.

# TODO: it is interesting to think about how can the featurizers be applied to 
# a sequence context. that is more tricky. think about CRFs and stuff like that.

# TODO: think about an easy map of neural network code to C code, or even to 
# more efficient python code, but that is tricky.

# think about when does something become so slow that it turns impractical?
# 

# TODO: look into how to use multi-gpu code.

# TODO: tools for masking and batching for sequence models and stuff.
# these are interesting. 

# for example, in the case of beam search, this would imply having a pad 
# sequence.

# check mongo db or something like that, and see if it is worth it.

# how would beam search look on tensorflow, it would require executing with the 
# current parameters, and the graph would have to be fixed.

# TODO: masking in general is quite interesting.

# possible to add words in some cases with small descriptions to the experiments.

# by default, the description can be the config name.

# the only thing that how approach changes is the way the model is trained.

# beam search only changes model training.

# is it possible to use the logger to safe information to generate results or 
# graphs.

# in tensorflow and torch, models essentially communicate through 
# variables that they pass around.

# for trading of computation and safety of predictions. 
# check that this is in fact possible. multi-pass predictions.

# TODO: an interesting interface for many things with state is the 
# step; and query functions that overall seem to capture what we care about.

# TODO: encode, decode, step, 
# I think that this is actually a good way of putting this interface.


# TODO: check what an efficient implementation in Tensorflow would look like.
# i.e., beam search.

# TODO: have something very standard to sweep learning rates. 
# what can be done here?

# TODO: rate annealing in ADAM can be useful.

# TODO: restart from the best previous point some model. 
# this is tricky to do correctly. how to hangle maintaining the model, 
# and choosing these things.

# Also, having the learning rate. this is interesting.

# TODO: review all the code and implement a new model.

# TODO: interesting techniques 

# TODO: guidelines for using some other project as a starting point 
# for research. check that you can get the performance mentioned in the paper.
# check that your changes did not change, often catatrophically, that you 
# can still get the performance in the paper.

# NOTE: it is not desirable for the rate counter to always return a step size, 
# because the optimzers have state which can be thrown away in the case of a 
# 

# NOTE: default aspects for training. what are the 
# important parts, like training until things stop decreasing.

# NOTE: writing deep learning code around the notion of a function.

# TODO: stuff to deal with unknown tokens.
# TODO: sort of the best performance of the model is when it works with 
# 

# there is all this logic about training that needs to be reused.
# this is some nice analysis there.

# TODO: notion of replay. do this for training environments.

# you have to get it out of x.

# NOTE: this is kind of like conditional execution.

# some of the stuff that I mention for building networks.

# TODO: construction with the help of map reduce.




## TODO: interface for encoder decoder models.

# TODO: stuff that handles LSTM masking and unmasking.

# NOTE: an interface that write directly to disk is actually 
# quite nice, 

# pack and unpack. functionality. keeping track of the batch dimension.

# use of map and reduce with hidden layers.

# TODO: perhaps the results can be put in the end of the model.

# padding seems to be done to some degree by the other model.

# TODO: types are inevitable. the question is can we do them without 
# being burdensome?

# NOTE: even just model checkpoints based on accuracy are a good way of going 
# about it.

# TODO: the notion of a replay settings. a set of variables that is kept track
# during various places.

# TODO: notion of driving training, and keeping some set of variables 
# around that allow me to do that.

# when some condition happens, I can execute a sequence of models.

# keeping the stuff way is a good way of not cluttering the output.

# TODO: notion of keeping enough information that allows me to reconstruct the 
# information that I care about.

# batching is quite important, and I need to do something that easily goes 
# from sequences to batches.

# there is a problem with the resets.
# once it resets, you have to consider where did it reset to, so you can 
# reduce the step size from there.
# basically, resetting gets much harder.

# TODO: basically switch between the format of column and row for sequences 
# of dictionaries. this is going to be simple. also, can just provide an 
# interface for this.

# TODO: checkpoints with resetting seem like a nice pattern to the model.

# TODO: extend this information about the model to make sure that the model
# is going to look the way

# it is going to be nice to extend the config generator to make sure that 
# things work out. for example, different number of arguments, or rather 
# just sequence multiple config generators or something like that.

# that is simple to do. it is just a matter of extending a lot of stuff.

# improve debbuging messages because this is hard to read.

# TODO: add some plotting scripts for typical things like training and 
# validation error. make sure that this is easy to do from the checkpoints and 
# stuff.

# TODO: perhaps for the most part, these models could be independent.
# this means that they would not be a big deal. this is going to be interesting.

# also, the notion of copying something or serializing it to disk is relevant.

# managing a lot of hyperparameters that keep growing is troublesome.
# might also be a good idea to use a tuner. 

# check on how to get a simple interface to 

# TODO: this is going 

# TODO: clean for file creation and deletion, this may be useful to group 
# common operations.

# TODO: registering various clean up operations and stuff like that.
# that can be done at teh 

# TODO: the replay of these things can be done skipping those that 
# were skipped, because those are eseentially wasted computation.
# there is an implied schedule from the replay.

# TODO: tests through config validation.

# TODO: possibility of running multiple times and getting the median.

# TODO: have stuff for pytorch.

# TODO: what is a good way of looking at different results of the model.
# it would be interesting to consider the case 

# TODO: those runs about rate patiences are the right way of doing things.
# TODO: add easy support for running once and then running much slower.

# TODO: rates make a large difference.

# TODO: add stuff that forces the reloading of all modules.

# TODO: equivalence between training settings.

# TODO: also, for exploration of different models. what is important to 
# check layer by layer.

# TODO: visualization of different parts of the model, and statistics. this is 
# going to be interesting. like exploring how different parts of the model. 
# change with training time.

# NOTE: things that you do once per epoch do not really matter, as they will be 
# probably very cheap computationally comparing to the epoch.

# TODO: ways of training the model with different ways.

# TODO: support for differential programming trained end to end.

# TODO: even to find a good configuration, it should be possible to figure 
# out reasonable configurations by hand.

# TODO: memory augmented with pointer networks.

# TODO: for DeepArchitect is going to be done through notion of functional
# programming.

# TODO: for hooks with forward and backward. 

# TODO: for DeepArchitect should be simple to 

# TODO: check that LSTM is going. check the sequence. variable length 
# packed sequence. what does it do?

# equivalence tests. this running multiple configs that you know that 
# should give the same result.

# TODO: it may be worth to come with some form of prefix code for the experiments
# it may be worth to sort them according to the most recent.

# TODO: using matplotlib animation can be something interesting to do.

# What kind of animation would be relevant. what are the interesting things 
# to animate.

# TODO: inteegrate attention mechanisms in DeepArchitect.

# TODO: some stuff to do an ensemble. it should be simple. just average the 
# predictions of the top models, or do some form of weighting.

# TODO: just analysing the mistakes is going to be interesting. also, it should
# be easy to train and validate on the true metrics. just change a bit deep 
# architect.

# it should be trivial to do, perhaps stack a few of these images, or tie
# the weights completely. maybe not necessary.

# TODO: processing the data is going to be interesting.

# throw some decent model at it. TODO: perhaps do a simple way of paralleizing
# MCTS with thompson sampling, or something like that. what should be 
# the posterior. it can be non-parametric, I guess, although it depends on 

# TODO: stuff for easily doing ensembles of models.

# TODO: multi gpu implementation, but this is mostly data parallel.

# TODO: add support for step size tuning.

# TODO: make io checkpointers that are a little smarter about when to save
# A model to disk. only save if it it has improved or it is going down.

# if just based on full predictions, that is fine, but if transitioning 
# requires running the model, it gets a bit more complicated. multi-gpu 
# options are possible, and doable. it is a matter of model serving.

# also, the multi-gpu aspect makes it possible.

# TODO: throw away 5 percent of the data where you make the most mistakes and 
# retrain. perhaps these are training mistakes.

# transpose a nested dict.

# TODO: several step functions to featurize, and perhaps visualize the 
# results of that model.

# TODO: add features for architectures easily.

# TODO: totally add stuff for cross validation.
# use the same train and test splits.
# TODO: perhaps base it around iterables.

# TODO: do train_dev split in a reasonable way.
# perhaps with indices. how did I do it in CONLL-2000 integrate that.

# TODO: for the seed add support for other things that may have independent 
# seeds.

# TODO: can add hyperparameters for the data loading too. it 
# it just get more complicated.

# TODO: in DeepArchitect stop as soon as things go to nan.
# this is going to be clear in the loss function.

# TODO: look at ways of making this work nicely. for the case of hyperparameter 
# tuning.

# TODO: add tools for the lightweight toolbox.

# TODO: perhaps add some installation information.

# TODO: also stuff to transfer between 