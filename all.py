
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
     
def retrieve_obj_vars(obj, var_names, fmt_tuple=False):
    return retrieve_values(vars(obj), var_names, fmt_tuple)

### partial application and other functional primitives
import functools

def partial_apply(fn, d):
    return functools.partial(fn, **d)

### dictionary manipulation
import pprint

def create_dict(ks, vs):
    assert len(ks) == len(vs)
    return dict(zip(ks, vs))

def merge_dicts(ds):
    out_d = {}
    for d in ds:
        for (k, v) in d.iteritems():
            assert k not in out_d
            out_d[k] = v
    return out_d

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

def retrieve_values(d, ks, fmt_tuple=False):
    out_d = {}
    for k in ks:
        out_d[k] = d[k]
    
    if fmt_tuple:
        out_d = tuple([out_d[k] for k in ks])
    return out_d

# TODO: difference between set and define.
# set if only exists.
# define only if it does not exist.
# both should have concrete ways of doing things.
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

def write_jsonfile(d, fpath):
    with open(fpath, 'w') as f:
        json.dump(d, f, indent=4)

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
        write_header=True, abort_if_different_keys=True):

    ks = tb.key_union(ds)
    assert (not abort_if_different_keys) or len( tb.key_intersection(ds) ) == len(ks)

    lines = []
    if write_header:
        lines.append( sep.join(ks) )
    
    for d in ds:
        lines.append( sep.join([str(d[k]) if k in d else '' for k in ks]) )

    tb.write_textfile(fpath, lines)

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
        date_s = "%s-%s-%s" % (d.year, d.month, d.day)
    time_s = ''
    if not omit_time:
        date_s = "%s:%s:%s" % (d.hour, d.minute, d.second)
    
    vs = []
    if not omit_date:
        vs.append(date_s)
    if not omit_time:
        vs.append(time_s)

    if len(vs) == 2 and time_before_date:
        vs = vs[::-1]        
    s = '|'.join(vs)

    return s

### useful iterators
import itertools

def iter_product(lst_lst_vals):
    return list(itertools.product(*lst_lst_vals))

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

# TODO: needs some error checking in case of no gpus. needs a try catch block.
def gpus_total():
    try:
        n = len(subprocess.check_output(['nvidia-smi', '-L']).strip().split('\n'))
    except OSError:
        n = 0
    return n

def gpus_free():
    return len(gpus_free_ids())

# NOTE: this is not just a matter about gpu utilization, it is also about 
# memory utilization.
# TODO: check if this handles memory appropriately.
# TODO: chec 
def gpus_free_ids():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', 
            '--format=csv,noheader'])
        perc_util = [float(x.split()[0]) for x in out.strip().split('\n')]

        ids = []
        for i, x in enumerate(perc_util):
            # NOTE: this is arbitrary for now. maybe a different threshold for 
            # utilization makes sense. maybe it is fine as long as it is not 
            # fully utilized.
            if x < 0.1:
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

# NOTE: this part here can be improved.
# NOTE: for now, this is just going to use a single gpu for each time. 

# doing double ssh to 

# TODO: check that you get the right prompt rather than something else.
# NOTE: certain things may be troublesome because of escaping the commands 
# correctly and stuff like that. I don't think that that would be 
# handled by itself.
# having that into account in the command passed may be a little troublesome.

# I need to handle this some how.
# there are cases where I can just run ssh directly.

### running jobs on the available computational nodes.
import inspect

def get_lithium_nodes():
    return {'gtx970' : ["dual-970-0-%d" % i for i in range(0, 13) ], 
            'gtx980' : ["quad-980-0-%d" % i for i in [0, 1, 2] ],
            'k40' : ["quad-k40-0-0", "dual-k40-0-1", "dual-k40-0-2"],
            'titan' : ["quad-titan-0-0"]
            }

# it is better to pass a user name.

# NOTE: prompting for password does not make much sense.
# gpu_memory_total
# gpu_memory_free
### TODO: add something to query how much memory is it 
# being used there.

# memory_total
# memory_free

# TODO: perhaps it is possible to run the model in such a way that makes 
# it easy to wait for the end of the process, and then just gets the results.
# for example, it puts it on the server, sends the command, and waits for 
# it to terminate. gets the data everytime.

# it has to connect for this...

# TODO: the question is which are the cpus that are free. 
# NOTE: it is a good idea to think about one ssh call as being one remote 
# function call.

# in the generation of calls lines, it is a good idea to see other aspects.

# also return the gpus that are free.

# TODO: should also be easy to check 

# it ssh for this.

# ignore the GPUs that are down.
# add functionality to use server if resources are available.

# what if something does not succeed, it still gives back something relevant.

## TODO: get to the point were I can just push something to the server 
# and it would work.

# TODO: perhaps it is possible to get things to stop if one of the 
# them is not up.
# I'm sure that this is going to fail multiple times.

# NOTE: this may need the toolbox there, otherwise, it is a pain.
def get_lithium_resource_availability(servername, username, password=None,
        abort_if_any_node_unavailable=True):

    # prompting for password if asked about.
    if password == None:
        password = getpass.getpass()

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
    resources = {}
    for gpu_type in nodes:
        resources[gpu_type] = []
        for host in nodes[gpu_type]:
            cmd = 'ssh -T %s python avail_resources.py' % host
            stdout_lines, stderr_lines = run_on_server(
                cmd, servername, username, password)

            # print stdout_lines, stderr_lines
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
                resources[gpu_type].append(d)
            else:
                assert not abort_if_any_node_unavailable
                resources[gpu_type].append(None)

    delete_script_cmd = 'rm avail_resources.py'
    run_on_server(delete_script_cmd, servername, username, password)
    return resources





# ideally, you just want to run things once.
# also, perhaps, I would like to just point at the folder and have it work.



### this has to be done better.

# read prompt for password_prompt


    # not quit because it is going to be one for each computer.

    # dictionary has to match there.


# I wonder if this works or not.

# it would be interesting to check if I can 


# get_resource_availability_on_lithium


# for the ones with a scheduler, it should be possible to do this easily without
# having to worry about querying for the resources.


# I will need to wrap this as a string. I need to be careful about it.

    # write the script once, and then remove it. be careful about this.

    # ssh -t negrinho@
    # "negrinho@128.2.211.186"

    # try to do it for a single one, and then run on it.
    # do not have my username in it.
    
    # go there, and print something 
    # there should exist code to distribute 
    # things easily.

    # NOTE: even just doing something like doing 
    # ssh once and getting all the data may be interesting
    # can I do that.

    # it is better if I can do it directly.

    # the laucnh script for a task needs to be easy.
    
    # just the creation the com
    pass

# NOTE: to get the available processors and gpus, it is necessary to be 
# careful about memory tool. I think 

# NOTE: may require a python multi line command. otherwise, just put in some 
# script and then erase it.

# NOTE: on lithium I have to write nohup things.
# running on lithium is kind of tricky. more sshs.
# NOTE: that a command is always to be ran on a specific node.
# I want to get information about this node. what is the problem, 
# not for every time

# TODO: develop functions to look at the most recent experiments, or the files
# that were change most recently.

# also stuff to look into the toolbox.
# the plug and play to make the experience 


def run_on_lithium(bash_command, servername, username, password=None, 
        num_cpus=1, num_gpus=0, folderpath=None, 
        wait_for_output=True, require_gpu_types=None, require_nodes=None,
        run_on_head_node=False):

        folderpath=None, wait_for_output=True, prompt_for_password=False

    if password == None and prompt_for_password:
        password = getpass.getpass()


# there is stuff that I need to append. for example, a lot of these are simple 
# to run. there is the syncing question of folders, which should be simple. 
# it is just a matter of  

    # depending on the requirements for running this task, different machines 
    # will be available. if you ask too many requirement, you may not have 
    # any available machines.

# I just don't want to do the logic to access it. also, perhaps create some file
# that I can reuse.

    
    
    ### do this later.



# # run_on_server(servername, bash_command, username=None, password=None,
# #         folderpath=None, wait_for_output=True, prompt_for_password=False)



# # # probably needs to exit if there are no available cpus.

# # # needs to figure out a gpu.
# # # it is more annoying.

# # I think that experiments should always be 

# #     script = ["",
#             # ]

#     run_on_server()

    # for requiring some gpu type, this is going to be a lot simpler.

    # TODO: this is mostly a matter of setting the right part of the model.
    # think about how to return the model.

    # basically, can move to one of these folders, put the script there, 
    # run it, and delete it.

# this is not true.

# submit a job there.

# NOTE: for running on matrix, there is the question about 
# this is for parallel parts of the model.

# whether to run on headnode

# can run multiple of these 

# def run_on_matrix(bash_command, servername, username, password=None, 
#         num_cpus=1, num_gpus=0, mem_budget=8.0, time_budget=60.0,
#         mem_units='gb', time_units='m', 
#         folderpath=None, wait_for_output=True, 
#         require_gpu_type=None, run_on_head_node=False):

#     if require_gpu_type is not None: 
#         raise NotImplementedError

#     # prompts for password if it has not been provided
#     if password == None:
#         password = getpass.getpass()

#     ## basically, needs slurm instructions.

#     'sbatch --cpus-per-task=%d --gres=gpu:%d --mem=%d' 
#     # matrix cmd has to have the job there.

#     run_on_server(matrix_cmd, **retrieve_values(locals(), 
#         ['servername', 'username', 'password', 'folderpath', 'wait_for_output'])


# # it depends if it is to run on the node or not.

# ####

#     if username != None: 
#         host = username + "@" + servername 
#     else:
#         host = servername
    
#     if password == None and prompt_for_password:
#         password = getpass.getpass()

#     if not wait_for_output:
#         bash_command = "nohup %s &" % bash_command

#     if folderpath != None:
#         bash_command = "cd %s && %s" % (folderpath, bash_command) 

#     sess = paramiko.SSHClient()
#     sess.load_system_host_keys()
#     #ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
#     sess.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     sess.connect(servername, username=username, password=password)
#     stdin, stdout, stderr = sess.exec_command(bash_command)
    
#     # depending if waiting for output or not.
#     if wait_for_output:
#         stdout_lines = stdout.readlines()
#         stderr_lines = stderr.readlines()
#     else:
#         stdout_lines = None
#         stderr_lines = None

#     sess.close()
#     return stdout_lines, stderr_lines




# commands are going to be slightly different.

# NOTE: leave this one for later.
# # this one is identical to matrix.
# def run_on_bridges(bash_command, servername, username, password, num_cpus=1, num_gpus=0, password=None,
#         folderpath=None, wait_for_output=True, prompt_for_password=False,
#         require_gpu_type=None):
#     raise NotImplementedError
#     pass

### NOTE: I shoul dbe able to get a prompt easily from the command line.

# NOTE: if some of these functionalities are not implemented do something.

# a way to run is to do a 

# this would work 

# echo "script" > run_slurm.sh && chmod +x run_slurm.sh && sbatch 
# (commands about resources) && rm run_slurm.sh 


# some of these are going to require creating a file.
# question about waiting for output.


# NOTE: if a file is required, this is suboptimal. I think that it should be 
# possible to run it on a remote machine. it can be a long command.


# this is probably going to be done in the right folder.

# two ssh and run a command.
# NOTE: how to cancel a session.
# 

# check if a file is on the server or not.

# have a sync files from the server.



### plotting 

# in the case of using the library in the server, this may fail.
# TODO: check if maplotlib is on the server or not.
try:
    import matplotlib.pyplot as plt 
except ImportError:
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

# TODO: these all need to be tested.
class PatienceRateSchedule:
    def __init__(self, rate_init=1.0e-3, rate_mult=0.1, rate_patience=5, 
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

class StopCounter:
    def __init__(self, stop_patience, minimizing=True):
        assert stop_patience >= 0

        self.stop_patience = stop_patience
        self.minimizing = minimizing
        
        self.counter = self.stop_patience
        self.prev_value = np.inf if minimizing else -np.inf
        self.num_steps = 0
    
    def update(self, v):
        assert self.counter > 0

        if (self.minimizing and v < self.prev_value) or (
                (not self.minimizing) and v > self.prev_value) :
            self.counter = self.rate_patience
        else:
            self.counter -= 1
        
        self.prev_value = v
        self.num_steps += 1

    def has_stopped(self):
        return self.counter == 0
    
    def reset(self):
        self.counter = self.stop_patience
        self.num_steps = 0

## TODO: implement the fix decrease schedule and stuff like that.
# commonly used in training architectures.

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

def keep_most_frequent(tk_to_cnt, num_tokens):
    top_items = sorted(tk_to_cnt.items(), 
        key=lambda x: x[1], reverse=True)[:num_tokens]
    return dict(top_items)

def index_tokens(tokens, start_index=0):
    assert start_index >= 0
    num_tokens = len(tokens)
    tk_to_idx = dict(zip(tokens, range(start_index, num_tokens + start_index)))
    return tk_to_idx

def reverse_mapping(a_to_b):
    b_to_a = dict([(b, a) for (a, b) in a_to_b.iteritems()])
    return b_to_a

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

# TODO: a class that takes care of managing the domain space of the different 
# architectures.

# this should handle integer featurization, string featurization, float
# featurization
# 

# features from side information.
# this is kind of a bipartite graph.
# each field may have different featurizers
# one field may have multiple featurizers
# one featurizers may be used in multiple fields
# one featurizer may be used across fields.
# featurizers may have a set of fields, and should be able to handle these 
# easily.

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

def create_runall_script_with_parallelization(exp_folderpath):
    fo_names = list_folders(exp_folderpath, recursive=False, use_relative_paths=True)
    # print fo_names
    num_exps = len([n for n in fo_names if path_last_element(n).startswith('cfg') ])

    ind = ' ' * 4
    # creating the script.
    sc_lines = [
        '#!/bin/bash',
        'if [ "$#" -ne 0 ] && [ "$#" -ne 2 ]; then',
        '    echo "Usage: run.sh [worker_id num_workers]"',
        '    exit 1',
        'fi',
        'if [ $# -eq 2 ]; then',
        '    worker_id=$1',
        '    num_workers=$2',
        'else',
        '    worker_id=0',
        '    num_workers=1',
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
        '        echo cfg$i',
        '        %s' % join_paths([exp_folderpath, "cfg$i", 'run.sh']),
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

# TODO: syncing folders

### project manipulation
def create_project_folder(folderpath, project_name):
    fn = lambda xs: join_paths([folderpath, project_name] + xs)

    create_folder( fn( [] ) )
    # typical directories
    create_folder( fn( [project_name] ) )
    create_folder( fn( ["data"] ) )
    create_folder( fn( ["experiments"] ) )    
    create_folder( fn( ["notes"] ) )
    create_folder( fn( ["temp"] ) )

    # code files (in order): data, preprocessing, model definition, model training, 
    # model evaluation, main to generate the results with different relevant 
    # parameters, setting up different experiments, analyze the results and 
    # generate plots and tables.
    create_file( fn( [project_name, "data.py"] ) )
    create_file( fn( [project_name, "preprocess.py"] ) )
    create_file( fn( [project_name, "model.py"] ) )    
    create_file( fn( [project_name, "train.py"] ) )
    create_file( fn( [project_name, "evaluate.py"] ) ) 
    create_file( fn( [project_name, "main.py"] ) )
    create_file( fn( [project_name, "experiment.py"] ) )
    create_file( fn( [project_name, "analyze.py"] ) )


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


    # list_folders(exp_folderpath)

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

# TODO: manipulation of the experiments to have precisely want I need.

# do the make csv file function.

### load all configs and stuff like that.
# they should be ordered.
# or also, return the names of the folders.
# it is much simpler, I think.

# stuff to deal with embeddings in pytorch.

# connecting two computational graphs in tensorflow.


# TODO: some functionality will have to change to reaccomodate
# the fact that it may change slightly.

### profiling

# TODO: figure out the right code to have here.

# csv from list of dicts
# filter 

# the maximum I can do is two dimensional slices.

# copying the code.

# stuff that I can use by running the toolbox.
# same thing on the compressing end. compress the file before downloading.

# generate csv from a list of dictionaries.

# this should be made easier.

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
# more than that, it start becomes a nuissance.

# top 10 plots for some of the most important ones.
# 

# need dynamic runs of the model, to make 

# perhaps can switch things around.



# some tools for 
# data.py da 
# train.py tr
# model.py mo
# preprocess.py pr 
# evaluate.py ev
# plot.py pl
# main.py ma
# experiments.py ex
# analyze an (maybe for extra things)
### ---
# notes
# data
# temp
# experiments
# analyses
# code

# get path

# maybe even write something to the model. this could be interesting.

# add stuff for feature extraction.



### TODO: upload to server should compress the files first and decompress them 
# there.

# may comment out if not needed.

# for boilerplate editing



# add these. more explicitly.
# TODO: add file_name()
# TODO: file_folder
# TODO: folder_folder()
# TODO: folder_name()

# for running scripts there are questions about waiting for termination
# and what not.

# add more information for the experiments folder.
# need a simple way of loading a bunch of dictionaries.

# there is the question of it is rerun or dry run.
# something like that.
# ideally, the person should decide if it want to capture or not.

# how many in parallel.
# the folder is going to be there and it will have some structure.

# let us run all the experiments just for fun.

# there is also the script for the experiment.

# how to profile.


# also needs a high level experiment to run all of them.

# basically, each on is a different job.
# I can base on the experiment folder to 
# with very small changes, I can use the data 
# data and the other one.
# works as long as paths are relative.

# can be done either through the sh files or through the config files 
# using information

# the names of these folders are going to be fixed.cnf cfg conf cfg 

# relative are better.

    # needs more information.

# maybe run experiments and stuff like that.

# each gets its runnable script or something like that.

# relative the projects.

# multiple experiments can be ran.
# 
# can create a config out of the elements used in the call. 
# 

# analyze the configs for the model.

# some number in parallel


# that can be part of the functionality of the research toolbox.

# as these 

# this model kind of suggests to included more information about the script 
# that we want to run and use it later.

## NOTE: can add things in such a way that it is possible to run it
# on the same folder. in the sense that it is going to change folders
# to the right place.

### and more.

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

# ignore the data for now.
# 

# stuff to kill all jobs. that can be done later.

# assumes that the experiment is ran in the experiment folder.
    

    # code folder path 
    # needs to substitute the function. it is annoying because it 

# remap some of the paths or something like that.
# absolute paths. 
# kind of tricky. what would that correspond to?

# it is tricky to make sure that this works.
### if it copies data and code, it needs to map stuff to


# is None... this is cool for getting the experiments to work.

### it means that I can run the code while being indexed to the part of the 
# model that actually matters.


# NOTE: this may have a lot more options about 
def run_experiment_folder(folderpath):
    pass

# there needs to exist some information about running the experiments. can 
# be done through the research toolbox.



### --data_folder /..../renato/Desktop/research/projects/beam_learn/data
# or it just uses some remapping

# assumes the code/ data 
# assumes that the relative paths are correct for running the experiment 
# from the current directory.

# this can be done much better, but for now, only use it in the no copying case.
# eventually, copy code, copy data depending on what we want to get out of it.
# it is going to be called from the right place, therefore there is no problem.


# in those ortho-iterators, it is perhaps a good idea to take those that 
# cmome next

# what to do with the output.

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

# something like this 
# run on remote gpu and cpu.
# TODO: run thsi locally or remotely. the gpu one takes more care.
def run_on_gpu():
    pass



def run_on_cpu():
    pass

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

# TODO: copy a version of the running code into the folder.

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

# some parts can be relaxed.

# check if I can get the current process pid.

# def run_cross_validation(train_matrix, train_vector, fold_generator_function,
#     fit_function, score_function):

#     # for each fold of the training data
#     score_list = list()
#     for train_fold_index, test_fold_index in fold_generator_function():
#         # train partition
#         train_fold_matrix = train_matrix[train_fold_index]
#         train_fold_vector = train_vector[train_fold_index]
#         # test partition
#         test_fold_matrix = train_matrix[test_fold_index]
#         test_fold_vector = train_vector[test_fold_index]

#         # train the model in training fold
#         fitted_model = fit_function(train_fold_matrix, train_fold_vector)

#         # score the fitted model on the test partition
#         fold_score = score_function(fitted_model, test_fold_matrix, test_fold_vector)
#         score_list.append(fold_score)

#     return np.mean(score_list)

# NOTE: can be done more generically.
# fit_functions, score_functions. that is it
# the fit and score already have the dataset there
# cross_validation is a single score...
# also needs to return the index of the scoring function.

# # NOTE in the score function, bigger is better
# def get_best_regularization_parameter(train_matrix, train_vector,
#     regularization_params_list, fold_generator_function, fit_function_factory,
#     score_function):

#     # best params for the model that we are training
#     best_params = None
#     best_score = - np.inf
#     for reg_params in regularization_params_list:
#         # NOTE that the parameters are a tuple
#         fit_function = fit_function_factory(*reg_params)

#         # compute the score according to cross validation
#         score = run_cross_validation(train_matrix, train_vector,
#             fold_generator_function, fit_function, score_function)

#         if score > best_score:
#             best_score = score
#             best_params = reg_params

#     return (best_params, best_score)

# have a way of sending all output to file.
# sending everything to files is a nice way of going about this.

# managing some experimental folders and what not.

# will need to mask out certain gpus.

# probably, it is worth to do some name changing and stuff.

# these are mostly for the servers for which I have access.

# need some function that goes over multiple machines.
# function that issues rtp.

# only run on certain gpus

# only check a subset of them.

# the running script for this is going to be much more high level.


# may need only the ones that are free.

# perhaps have something where it can be conditional on some information 
# about the folder to look at.
# If it is directly. it should be recursive, and have some 
# form of condition. I need to check into this.

# totally allows you to do what you want

# information on what gpu it was ran on. 
# check if some interfacing with sacred can be done.

# call these these functions somewhere else. 
# this should be possible.

# for interacting with slum.
# all the s(something commands.)
# for getting information about the server.


# do some well defined units;
# have some well defined server names.

# psc, lithium, russ', matrix

# creates the wrappers automatically.

# needs to be updated.

# folder creation and folder management.

# mostly for reading and writing, since I think it is always the same.
# I think that it should get a single figure for it.
# depending on what you pass, you should be able to get what you want.

# also, some high level adjustment functions

# capture all output and make it go to the 

# I don't think that I will need it.

# --- something like this 
# run_on_gpu(host_name, server_name):
# run_on_cpu():

# some can be done through POpen or something like that.

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

# stuff to get the git id.


# in some cases, operate directly on files.
# take a list of files, and return a csv.

# wrap certain metrics to make them easy to use.

# prefer passing a dictionary, and then unpacking.

# something for creating paths out of components

# TODO: stuff for using Tensorflow and PyTorch and Tensorboard.

# add stuff for json. it may be simpler to read.

# to do remote calls, it seems fairly simple. just use the subprocess part of the 
# model.

# also, with some form of namespaces for model.

# choose whether to print to the terminal or to a file.
# this can be done in the memer. 
# I think that a single line is enough to get this working.

# code for running stuff remotely may be interesting.

# better dictionary indexing and stuff like that.

# that can be done in a more interesting way.

# there are probably some good ways of getting the configurations.
# ideally, I think that it should be possible to sample configurations 
# uniformly.


# this should be something easy to do. 
# it can be done with choose.

# logging can be done both to a file and to the terminal.

### TODO: maybe not all have to be collapsed to the lower level.
# NOTE: not throughly tested.


# iter_product([1,2,3], [2,3,4])
# iter_product({'a' : [1,2,3], 'b' : [2,3,4]}) => directly to something.

# change some number of pairs.

# jio

# what if they are named.

# may be positional or not.
# does all the other ones, 

# stuff for controlling randomness.

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

# stuff to copy directories and so forth.

# use more consistent spelling

# add common ways of building a command line arguments.

# add common arguments and stuff like that.
# for learning rates, and step size reduction.

# for dealing with subsets, it is important to work with the model.

# come up with a special format for main.

# do stuff to deal with json. it may be easier to do actually.

# do some structures for keeping track of things. easily.
# perhaps just passing a dictionary, and appending stuff.
# dealing with summary taking it is difficult.

# files to write things in the CONLL-format 

# stuff for step tuning, and stuff like that. when to decrease, 
# what to assume

# there should exist stuff that ignores if models are iterable or not.

# I'm making wide use of Nones. I think that that is fine.
# keep this standard, in the sense that if an option is not used,
# it is set to None.

# stuff to make or abort to create a directory.

# iterate over all files matching a pattern. that can be done easily with re.

# # TODO: still have to do the retraining for CONLL-2003

# an easierway of doing named tuples with the same name of the variable.
# 

# simple scripts for working with video and images.

# map values. that is something that can be done over the dictionary.

# the whole load and preprocessing part can be done efficiently, I think 
# I think that assuming one example per iterable is the right way of going 
# about this. also, consider that most effort is not in load or preprocessing
# the data, so we can more sloppy about it.
    

# NOTE: there are also other iterators that are interesting, but for now,
# I'm not using them. like change of all pairs and stuff like that.


# set_seed
# and stuff like that. 
# it is more of an easy way of doing things that requires me to think less.


# from lower level things to more higher level things. 
# like construct from data and preprocess.
# lump together common operations.

# copy directory is important
# as it is copy file.

# copy directory. all these can be done under the ignore if it exists 
# or something like that. I have to think about what is the right way of 
# going about it.

# accumulation, keeping track of statistics.
# like, do it once, and then push to the model.
# it is cleaner.

# random permutation and stuffle.

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

# retrieve("num_words_keep", )
# this is done from the local context.

# functions to send emails upon termination. that is something interesting,
# and perhaps with some results of the analysis.

# also, it is important to keep things reproducible with respect to a model.
# some identification for running some experiment.

# plotting is also probably an interesting idea.


# also, potentially add some common models, and stuff like that.


# have an easy way of generating all configurations, but also, orthogonal 
# configurations.

# and also, pairs of configurations.

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

# easier to apply. less code.

# most of it is training configuration.

# around 5000 lines of toolbox code is OK.

# essentially, question about using these paths, vs using something that is more 
# along the lines of variable arguments.

# working on the toolbox is very reusable.

# more like aggregated summary and stuff like that.

# it is tedious to add one more argument and stuff like that.
# keep only the model.

# especially the functions that take a large number of arguments are easier 
# to get them working.

# featurizers and feature representations and stuff like that.
# something that you can apply once, 

# check if PyTorch keeps sparse matrices too? probably not.

# stuff that is task based, like what wrappers and loaders, and preprocessors
# and stuff like that.

# news model.
# think about the model.

# stuff for image recognition, image captioning, visual question answering

# all these setups still stand.

# probably has to interact well with dictionaries. for example, I could 
# pass an arbitrary dictionary and it would still work.
# also should depend on the sequence, 

# and it may return a dictionary or a tuple. that seems reasonable enough.

# maybe the separators can be different. 
# maybe no spaces allowed and other separators not allowed too.

# manipulation of text files is actually quite important to make this practical.
# tabs are not very nice to read.

# allow simple 

# NOTE: going back and forth using these dictionaries is clearly something that 
# would be interesting to do easily.

# testing the 

# some capture of outputs to files.

# remember than things are always locally.

# something to sample some number uniformly at random from the cross 
# product.
# it is simple to construct the list from the iterator.
# 

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

# small toolbox of little things.

# logging is going to be done through printing.

# either returns one of these dictionaries or just changes it.

# be careful about managing paths. use the last one if possible, t

# stuff for string manipulation

# the summaries are important to run the experiments.

# simple featurizers for text, images, and other stuff.

# dictionaries are basically the basic unit into the model.

# another important aspect is the toolbox aspect, where you 
# things are more consistent that elsewhere

# TODO: fix the gpu code in case there are no gpus.
# this code needs to have a lot more exception handling in some cases where 
# it makes sense.

# creating command-line argument commands easily

# TODO: more stuff to manage the dictionaries.

### another important aspect.
# for creating interfaces.
# actual behavior of what to do with the values of the arguments
# is left for the for the program.
# use the values in 

# may specify the type or not.


    # NOTE: may there is a get parser or something like that.

# maybe add an option that is simp

# easy to add arguments and stuff like that. what 
# cannot be done, do it by returning the parser.

# may be more convenient inside the shell.

# this works. gets trickier 

# run on machine is mostly useful for remote machines, therefore 
# you should should think about running things in the command line.
# not Python commands.

# def run_remotely():
#     pass

# # get the output in the form of a file. that is good.

# get all available machines 
# TODO: this may be interesting to do at the model.

# this is especially useful from the head node, where you don't have to 
# login into machines.

# getting remote out and stuff like that.

# run on cpu and stuff like that.
# depending on the model.

# even for the 

# can also, run locally, but not be affected by the result.

# it works, it just works.

### some simple functions to read and write.

# get current commit of the model.

# stuff to handle dates.


# if it is just a matter of tiling the graph, it can be done.

# can either do it separately r 


# GridPlots and reference plots.

# all factors of variation.

# class ReferencePlot:
#     pass
# the goal is never have to leave the command line.

# scatter plots too. add these.

# folder creation

# tests if already exists.
#  some useful manipulation for dictionaries.

# TODO: generating a plot for a graph.
# TODO: generating code for a tree in tikz.

### for going from data to latex

# NOTE: no scientific notation until now.
# possibilities to add, scientific notation.
# to determine this, and what not.

# say which entries to bold, or say to bold the largest element, smallest 
# in line or not.
# something with an header and a tail.

# may change if I add an omission mask, but that is more complex. 
# not needed for now.

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

# doing everything in json makes it more portable.

# ignoring hidden folders is different than ignoring hidden files.
# add these two different options.

# perhaps ignoring hidden files vs ignoring hidden folders make more sense.

# partial application for functions is also a possibility.

# potentially see paramiko for more complex operations.

# are there better ways of doing this?
# perhaps.

### I think that it has enough body to go to some a common space.

# can I do anything with remote paths to run things. I think that it may 
# be simpler to run models directly using other information.

# also, from dictionary.

# if no destination path, in the download, it assumes the current folder.
# TODO: always give the right download resulting directory.

# probably more error checking with respect to the path.

# some of these questions are applicable to creation and deletion of repos.

# also, directly using ways of keeping the results. some auxiliary function 
# for manipulating and for storing those values in the object directly.
# this is interesting, because it is simple to do.

# maybe this is an ssh path rather than a general remote path. 
# maybe can add an output format.

# perhaps, I should check that I'm in fact looking at a directory.

# TODO: perhaps will need to add ways of formating the json.

# NOTE: add support for nested columns.

# add support for grids of images.

# research_toolbox

# stuff for profiling and debugging.

# should be easy to compute statistics for all elements in the dataset.

# also, adopt a more functional paradigm for dealing with many things.
# all things that I can't do, will be done in way  


# using the correct definition is important. it will make things more consistent.

# even for simpler stuff where from the definition it is clear what is the 
# meaning of it, I think that it should be done.

# for files f

# don't assume more than numpy.
# or always do it through lists to the extent possible.

# use full names in the functions to the extent possible

# use something really simple, in cases where it is very clear.

### this one can be used for the variables.

# perhaps set local vars.


### extremely useful for delegation. for delegation and for passing around 
# information. positional arguments are hard to keep track of.
# easier to 

# json is the way to go to generate these files.

# decoupling how much time is it going to take. I think that this is going to 
# be simple.

### NOTE: this can be done both for the object and something else.

# it they exist, abort. have flag for this. for many cases, it make sense.

# important enough to distinguish between files and folders.

# perhaps better to utf encoding for sure.
# or maybe just ignore.

# for download, it also makes sense to distinguish between file and folder.

# aggregate dictionaries. that should be nice.

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

# to use srun and stuff.

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

# typical run, dump the arguments to a file, 
# dump the results to a file.
# probably should have a way of indexing the same results.

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
# just 

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

# getting to a point where you can run one experiment by calling a 
# bash script with 32 arguments is what you want to do. after that, 
# it is quite trivial to parallelize

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

# how to keep track of the changes.

# multiple runs for the same experiment? 
# copy some experiment configuration? sounds hard...
# better not.

# merging two experiments.

# some tools for 
# data.py da 
# train.py tr
# model.py mo
# preprocess.py pr 
# evaluate.py ev
# plot.py pl
# main.py ma
# experiments.py ex
# analyze an (maybe for extra things)
### ---
# notes
# data
# temp
# experiments
# analyses
# code

# get paths to the things that the model needs. I think that it is a better way of
# accomplishing this.

# analysis is typically single machine. output goes to something.

# copy_code.

# code in the experiment.
# stuff like that.

# for getting only files at some level
# 

# do the groupby in dictionaries
# and the summaries.

# should be simple to use the file iterator at a high level.
# produce the scripts to run the experiment.

# decide on the folder structure for the experiments... maybe have a few
# this is for setting up the experiments.

# I'm making a clean distinction betwenn the files that I want to run and 
# the files that I want to get.

# also, I should have some simple machine names.
# for example, for the ones in Alex server.

# stuff to process some csv files.
# stuff to manipulate dates or to sort according to date. 
# perhaps create datetime objects from the strings.

# the question is sorting and organizing by the last run.
# it is also a question of 

# will probably need to set visability for certain variables first.

# the goal is to run these things manually, without having to go to the 
# server.

# just keeping the password somewhere are look it up.

# most of the computational effort is somewhere else, so it is perfectly fine 
# to have most of the computation be done somewhere else.

# have a memory folder, it is enough to look for things. 
# can create experiments easily.

# having a main.
# it is possible to generate an experiment very easily using the file.
# and some options for keeping track of information.

# this can be done to run the code very easily.
# taking the output folder is crucial for success.

# run this with different values of the configuration.

# if they don't have the same, I can add Nones.

# list of dictioaries to CSV.

# experiment which is simply some identification.
# counting.

# /foldername/exp[n]/files
# perhaps readme.txt
# code also, and perhaps data.
# the experiments should be created with the path of the folder in 
# mind.

# path to the main folder
# also, some list of values for the parameters.
# it takes some number of configurations for the parameters.
# let us say that it does not repeat the name of the thing multiple 
# times.

# (.....) # ordered.

# 

# if the code is there, there shouldn't exist really a problem with getting it to 
# work.

# this may have a lot of options that we want to do to get things to 
# work. maybe there is some prefix and suffix script that has to be ran for these
# models. this can be passed somewhere.


### TODO: other manipulation with the experiment folder.
# that can be done.


# it should copy the code without git and stuff. this is going to be 
# pain to get right.

# there are folders and subfolders. 

# may exist data_folderpath or something, but I would rather pass that 
# somewhere else. also, maybe need to change to some folder.

# TODO: probably do something to add requirements to the code. 
# this would be something interesting.


# run one of these folders on the server is actually quite interesting.
# I just have to say run folder, but only every some experiment.
# that is what I need.

# probably needs more interface information.
# 

# operating slurm form the command line.

# this is going to be important for matrix and the other ones.

# sorted, with stable keys.

# important to request the right amount of data and stuff like that.

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

# allocation of gpus and stuff like that 
# this is going to be tricky.

# available memory per gpu and stuff like that.

# probably I should have some way of asking for resources.

# maybe wrap in a function.
# gpu machines in lithium.


# copying the code is a good idea for now.


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

# the path should be relative or something like that.
# the configuration should be easy to do.

# NOTE: I will probably ignore the fact that 

# may just find a new name.

# always ask for a name.

# simple naming conventions
# typical folder would be experiment.

# can be local or it can be global
# run the folder.
# 

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

# some number of different jobs.

# split then in different jobs.

# data <- if we choose to copy it.
# code <- if we choose to copy it.
# experiments
# readme.txt
# jobs <- all the jobs to run the experiments.

# do it all in terms of relative paths.
# or perhaps map them to non-relative paths.

# can add more configurations to the experiments folder.

# assumes that the config carries all the necessary information.

# assumes that the model knows what to do with respect to the given data path.

# NOTE: capturing the output could be done at the level of the process.
# that would be the best thing that we could do

# the question is how much can we get.

# it is a question of reentry and checkpoints and stuff. it runs 
# until it does not need to run anymore.
# perhaps keep the model.

# stuff that I can fork off with sbatch or something.

# get a subset of files from the server.
# that is quite boring to get all the information from the files.

# there is stuff about making parents and things like that.

# the question is the type of run that we need and stuff like that
# can I run a script that forks some number of jobs.
# or can I have a script that run some fraction of the model

# run.sh 0 .... njobs
# run.sh all

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

# launching the job can still be done by logging in in most cases.
# I think that it is not too bad.

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
# 

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

# TODO: try the alternative that they suggested.

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

# do things for the exploration of csv files. 
# this is going to be very handy to explore the results.

# easy way of fixing some of the arguments.

# NOTE: some of the stuff for connecting graphs in tensorflow.

# TODO: once it grows to a certain range, it will be useful to create toolbox 
# files mimicking some of the evaluation metrics that we looked at.
# this makes sense only after reaching a certain number of things to 
# do, or having libraries that are not guaranteed to exist in the model.

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

# make it easy to generate custom scripts to the 

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

# note, the parallel run script may not work there in the serverjk:w

# TODO: I will need some form of managing experiments across projects.jk:w

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
#

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

# it would  be interesting to add something  

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
# 

# make it easy to sync the folders. how?
# I think that for now, the most important is syncing everything.

# TODO: perhaps sync remote folder through the command line.
# it is sort of a link between two folders.

# I don't think that the current one is correct.

# find a nice way of keeping track of the experiments having a certain 
# configuration. it helps thinking about what I am doing.

# NOTE: what is the advantage of using scp or rsync

# TODO: use cases about making things work. 

# sync two folders or multiple folders

# --force-rerun option.
# --skip if 

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

# handle properly the non-existence of visualization services.

# NOTE: it is  easy to define a function locally and then update it to the
# server.

# testing code for interacting with the server is nice because I can stop if 
# it fails to reach it, so I can iterate fast.

# NOTE: it mostly makes sense to get stuff from the server to the 
# local host.

# TODO: add function to flatten a dictionary.
# and perhaps functions to go in the other directions.

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