
### auxiliary functions for avoiding boilerplate in in assignments
def retrieve_local_vars(var_names, fmt_tuple=False):
    return retrieve_values(locals(), var_names, fmt_tuple=fmt_tuple)

def define_obj_vars(obj, d):
    for k, v in d.iteritems():
        assert not hasattr(obj, k) 
        setattr(obj, k, v)
     
def retrieve_obj_vars(obj, var_names, fmt_tuple=False):
    return retrieve_values(vars(obj), var_names, fmt_tuple)

### dictionary manipulation
import pprint

def create_dict(ks, vs):
    return dict(zip(ks, vs))

def merge_dicts(ds):
    out_d = {}
    for d in ds:
        for (k, v) in d.iteritems():
            assert k not in out_d
            out_d[k] = v
    return out_d

def retrieve_values(d, ks, fmt_tuple=False):
    out_d = {}
    for k in ks:
        out_d[k] = d[k]
    
    if fmt_tuple:
        out_d = tuple([out_d[k] for k in ks])
    return out_d

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

# check if files are flushed automatically upon termination, or there is 
# something else that needs to be done. or maybe have a flush flag or something.
def capture_output(f, capture_stdout=True, capture_stderr=True):
    """Takes a file as argument. The file is managed externally."""
    if capture_stdout:
        sys.stdout = f
    if catpure_stderr:
        sys.stderr = f

### directory management.
import os
import shutil

# TODO: probably needs further clarification on if it is a folder or a file.
def path_prefix(path):
    return os.path.split[0]

def path_last_element(path):
    return os.path.split[1]

def path_relative_to_absolute(path):
    return os.path.abspath(path)

def file_exists(path):
    return os.path.isfile(path)

def folder_exists(path):
    return os.path.isdir(path)

def path_exists(path):
    return os.path.exists(path)

def create_directory(path):
    os.makedirs(path)

# TODO: change to the correct notation, be consistent.
# TODO: ..., also, perhaps if it does not exist.
def copy_file(src_path, dst_path):
    assert file_exists(src_path)

    if folder_exists(dst_path):
        pass

def copy_folder(src_path, dst_path, ignore_hidden=False):
    pass

def delete_folder(path, abort_if_nonempty=True, recursive=False):
    pass

# NOTE: should abort if not a file.
def delete_file(path):
    pass

# TODO: check that this is in fact correct. add comment about behavior.
def list_paths(folderpath, ignore_files=False, ignore_dirs=False, 
        ignore_hidden_dirs=True, ignore_hidden_files=True, ignore_file_exts=[], 
        recursive=False):
    assert folder_exists(folderpath)
    
    path_list = []
    # enumerating all desired paths in a directory.
    for root, dirs, files in os.walk(folderpath):
        if ignore_hidden_dirs:
            dirs[:] = [d for d in dirs if not d[0] == '.']
        if ignore_hidden_files:
            files = [f for f in files if not f[0] == '.']
        files = [f for f in files if not any([
            f.endswith(ext) for ext in ignore_file_exts])]

        if not ignore_files:
            path_list.extend([join_paths([root, f]) for f in files])
        if not ignore_dirs:
            path_list.extend([join_paths([root, d]) for d in dirs])
        if not recursive:
            break
    return path_list

def list_files(folderpath, 
        ignore_hidden_dirs=True, ignore_hidden_files=True, ignore_file_exts=[], 
        recursive=False):

    args = retrieve_vars(['recursive', 'ignore_hidden_dirs', 
        'ignore_hidden_files', 'ignore_file_exts'])

    return list_paths(folderpath, ignore_dirs=True, **args)

def list_folders(folderpath, 
        ignore_hidden_files=True, ignore_hidden_dirs=True, ignore_file_exts=[], 
        recursive=False):

    args = retrieve_vars(['recursive', 'ignore_hidden_dirs', 
        'ignore_hidden_files', 'ignore_file_exts'])
    return list_paths(folderpath, ignore_files=True, **args)

def join_paths(paths):
    return os.path.join(*paths)

def pairs_to_filename(ks, vs, kv_sep='', pair_sep='_', prefix='', suffix='.txt'):
    pairs = [kv_sep.join([k, v]) for (k, v) in zip(ks, vs)]
    s = prefix + pair_sep.join(pairs) + suffix
    return s

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

# TODO: maybe add units. (in init?)
class TimeTracker:
    def __init__(self):
        self.init_time = time.time()
        self.last_registered = self.init_time
    
    def secs_since_start(self):
        now = time.time()
        elapsed = now - self.init_time
        
        return elapsed
        
    def secs_since_last(self):
        now = time.time()
        elapsed = now - self.last_registered
        self.last_registered = now
        
        return elapsed            

# TODO: maybe add units. (in init?)
class MemoryTracker:
    def __init__(self):
        self.last_registered = 0.0
        self.max_registered = 0.0

    def mbs_total(self):
        mem_now = mbs_process(os.getpid())
        if self.max_registered < mem_now:
            self.max_registered = mem_now
        
        return mem_now
        
    def mbs_since_last(self):
        mem_now = self.mbs_total()        
        
        mem_dif = mem_now - self.last_registered
        self.last_registered = mem_now
        
        return mem_dif

# TODO: maybe add units. (in init?) same for the other below.
def print_time(timer, pref_str=''):
    print('%s%0.2f seconds since start.' % (pref_str, timer.secs_since_start()))
    print("%s%0.2f seconds since last call." % (pref_str, timer.secs_since_last()))

def print_memory(memer, pref_str=''):
    print('%s%0.2f MB total.' % (pref_str, memer.mbs_total()))
    print("%s%0.2f MB since last call." % (pref_str, memer.mbs_since_last()))
    
def print_memorytime(memer, timer, pref_str=''):
    print_memory(memer, pref_str)
    print_time(timer, pref_str)    

def print_oneliner_memorytime(memer, timer, pref_str=''):
    print('%s (%0.2f sec last; %0.2f sec total; %0.2f MB last; %0.2f MB total)'
                 % (pref_str,
                    timer.secs_since_last(), timer.secs_since_start(),
                    memer.mbs_since_last(), memer.mbs_total()))

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
        return dict(d)

class ConfigDict(ArgsDict):
    pass

class ResultsDict(ArgsDict):
    pass

### command line arguments
import argparse

class CommandLineArgs:
    def __init__(self, args_prefix=''):
        self.parser = argparse.ArgumentParser()
        self.args_prefix = args_prefix
    
    def add(self, name, type, default=None, optional=False, help=None,
            valid_choices=None, list_valued=False):
        valid_types = {'int' : int, 'str' : str, 'bool' : bool, 'float' : float}
        assert type in valid_types

        nargs = 1 if not list_valued else '+'
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
    return len(subprocess.check_output(['nvidia-smi', '-L']).strip().split())

def gpus_free():
    return len(gpus_free_ids())

def gpus_free_ids():
    return subprocess.check_output(['nvidia-smi' '--query-gpu=utilization.gpu', 
        '--format=csv,noheader'])

def gpus_set_visible(ids):
    n = gpus_total()
    assert all([i < n and i >= 0 for i in ids])
    subprocess.call(['export', 'CUDA_VISIBLE_DEVICES=%s' % ",".join(map(str, ids))])

### download and uploading
import subprocess
import os

def remote_path(username, servername, path):
    return "%s@%s:%s" % (username, servername, path)

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

# TODO: the recursive functionality is not implemented. check if it is correct.
# needs additional checking for dst_path. this is not correct in general.
def download_from_url(url, dst_path, recursive=False):
    subprocess.call(['wget', url])
    filename = url.split('/')[-1]
    os.rename(filename, dst_path)

### parallelizing on a single machine
import psutil
import multiprocessing
import time

def mbs_process(pid):
    psutil_p = psutil.Process(pid)
    mem_p = psutil_p.memory_info()[0] / float(2 ** 20)
    
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

def run_on_server(servername, bash_command, username=None, password=None,
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
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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

### plotting 
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

### TODO: not finished
def uniform_sample_product(lst_lst_vals, num_samples):
    for i in xrange(num_samples):
        pass

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

### remote function calls
# import paramiko

# probably will have to install on the server.


### finish the scripts to run multip

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

# some integration with scared is necessary.

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


# add sample uniformly

## 

# this should be something easy to do. 
# it can be done with choose.

# sample from a 

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

# common iterators.
# 

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

# HOST = "negrinho@128.2.211.186"
# COMMAND = "ls"

# def download_from_server(username, servername, src_path, dst_path, recursive=False):


# ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
#                        shell=False,
#                        stdout=subprocess.PIPE,
#                        stderr=subprocess.PIPE)
# result = ssh.stdout.readlines()

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

# be consistent about hidden dirs vs hidden folders.

# perhaps ignoring hidden files vs ignoring hidden folders make more sense.

# partial application for functions is also a possibility.

# and perhaps. 

# potentially see paramiko for more complex operations.

# are there better ways of doing this?
# perhaps.

### I think that it has enough body to go to some a common space.

# can I do anything with remote paths to run things. I think that it may 
# be simpler to run models directly using other information.

# also, from dictionary.

# if no destination path, in the download, it assumes the current folder.
# 

# probably more error checking with respect to the path.

# some of these questions are applicable to creation and deletion of repos.

# also, directly using ways of keeping the results. some auxiliary function 
# for manipulating and for storing those values in the object directly.
# this is interesting, because it is simple to do.

# maybe this is an ssh path rather than a general remote path. 
# maybe can add an output format.

# perhaps, I should check that I'm in fact looking at a directory.

# TODO: function to merge dictionaries.

# TODO: perhaps will need to add ways of formating the json.

# TODO: also, simple ways of reading the file.

# TODO: something to set an object.

# directories --- dirs --- folders.

# folder vs file.

# NOTE: add support for nested columns.

# add support for grids of images.

# research_toolbox

# stuff for profiling and debugging.

# should be easy to compute statistics for all elements in the dataset.

# also, adopt a more functional paradigm for dealing with many things.
# all things that I can't do, will be done in way  


# be consistent in the use of path. like folder_path, file_path, path
# you can have both. 

# src and dst seem reasoanble.

# using the correct definition is important. it will make things more consistent.

# even for simpler stuff where from the definition it is clear what is the 
# meaning of it, I think that it should be done.

# for files f

# don't assume more than numpy.
# or always do it through lists to the extent possible.

# use full names in the functions to the extent possible

# ComandLineArguments()

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

# have a way to setup things in the model.

# use required 

# running different configurations may correspond to different results.

# useful to have a dictionary that changes slightly. simpler than keeping 
# all the argumemnts around.
# useful when there are many arguments.

# run the model in many differetn 

# transverse all folders in search of results files and stuff like that.
# I think path manipulation makes a lot more sense with good tools.
# for total time and stuff like that.

# to use srun and stuff.

# run a remote command and stuff.

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
# 

    

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