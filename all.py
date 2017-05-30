
### auxiliary functions for avoiding boilerplate in in assignments
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

# TODO: difference between set and define.
# set if only exists.
# define only if it does not exist.
# both should have concrete ways of doing things.
def set_values(d_to, d_from):
    raise NotImplementedError

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

def create_folder(folderpath, 
        abort_if_exists=True, create_parent_folders=False):

    assert not file_exists(folderpath)
    assert create_parent_folders or folder_exists(path_prefix(folderpath))

    if abort_if_exists:
        assert not folder_exists(folderpath)
    os.makedirs(folderpath)

def copy_file(src_filepath, dst_filepath, 
        abort_if_dst_exists=True, create_parent_folders=False):

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

    assert file_exists(src_folderpath)
    assert src_folderpath != dst_folderpath
    assert not (abort_if_dst_exists and folder_exists(dst_folderpath))  

    if (not abort_if_dst_exists) and folder_exists(dst_folderpath):
        delete_folder(dst_folderpath, abort_if_nonempty=False)
    
    pref_dst_fo = path_prefix(dst_folderpath)
    assert create_parent_folders or folder_exists(pref_dst_fo)
    create_folder(dst_folderpath, create_parent_folders=create_parent_folders)
    
    # create all folders in the destination.
    args = retrieve_values(locals(), ['ignore_hidden_folders', 'ignore_hidden_files'])  
    fos = list_folders(src_folderpath, use_relative_paths=True, recursive=True, **args)

    for fo in fos:
        fo_path = join_path([dst_folderpath, fo])
        create_folder(fo_path, create_parent_folders=True)

    # copy all files to the destination. 
    fis = list_files(src_folderpath, use_relative_paths=True, recursive=True, **args)
        
    for fi in fis:
        src_fip = join_paths([src_folderpath, fi])
        dst_fip = join_path([dst_folderpath, fi])
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
            files = [f for f in files if not any([
                f.endswith(ext) for ext in ignore_file_exts])]

        if not ignore_files:
            if not use_relative_paths:
                path_list.extend([join_paths([root, f]) for f in files])
            else: 
                path_list.extend(files)
        if not ignore_dirs:
            if not use_relative_paths:
                path_list.extend([join_paths([root, d]) for d in dirs])
            else: 
                path_list.extend(dirs)
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
    
    def time_since_start(self, units='s'):
        now = time.time()
        elapsed = now - self.init_time
        
        return convert_between_time_units(elapsed, dst_units=units)
        
    def time_since_last(self, units='s'):
        now = time.time()
        elapsed = now - self.last_registered
        self.last_registered = now

        return convert_between_time_units(elapsed, dst_units=units)            

# TODO: maybe add units. (in init?)
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

# TODO: maybe add units. (in init?) same for the other below.
def print_time(timer, pref_str=''):
    print('%s%0.2f seconds since start.' % (pref_str, timer.time_since_start()))
    print("%s%0.2f seconds since last call." % (pref_str, timer.time_since_last()))

def print_memory(memer, pref_str=''):
    print('%s%0.2f MB total.' % (pref_str, memer.memory_total()))
    print("%s%0.2f MB since last call." % (pref_str, memer.memory_since_last()))
    
def print_memorytime(memer, timer, pref_str=''):
    print_memory(memer, pref_str)
    print_time(timer, pref_str)    

def print_oneliner_memorytime(memer, timer, pref_str=''):
    print('%s (%0.2f sec last; %0.2f sec total; %0.2f MB last; %0.2f MB total)'
                 % (pref_str,
                    timer.time_since_last(), timer.time_since_start(),
                    memer.memory_since_last(), memer.memory_total()))

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

# there are also questions about the existence of the path or not.
# also with the path manipulation functionality. as a last resort, I could 
# just delegate to the function that we call.

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
# ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
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

def create_run_script(project_folderpath, main_relfilepath,
        argnames, argvals, out_filepath):
    
    # creating the script.
    indent = ' ' * 4
    sc_lines = [
        '#!/bin/bash',
        'set -e',
        'cd %s' % project_folderpath, 
        'python %s \\' % join_paths([project_folderpath, main_relfilepath])
        ]
    sc_lines += ['%s--%s %s \\' % (indent, k, v) 
        for k, v in it.izip(argnames[:-1], argvals[:-1])]
    sc_lines += ['%s--%s %s' % (indent, argnames[-1], argvals[-1]), 
                 'cd -']
    write_textfile(out_filepath, sc_lines, with_newline=True)
    # giving run permissions.
    st = os.stat(out_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(out_filepath, st.st_mode | exec_bits)

# could be done better with a for loop.
def create_runall_script(exp_folderpath):
    fo_names = list_folders(exp_folderpath, recursive=False, use_relative_paths=True)
    num_exps = len([n for n in fo_names if n.startswith('cfg') ])

    # creating the script.
    sc_lines = ['#!/bin/bash']
    sc_lines += [join_paths([exp_folderpath, "cfg%d" % i, 'run.sh']) 
        for i in xrange(num_exps)]

    # creating the run all script.
    out_filepath = join_paths([exp_folderpath, 'run.sh'])
    print out_filepath
    write_textfile(out_filepath, sc_lines, with_newline=True)
    st = os.stat(out_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(out_filepath, st.st_mode | exec_bits)

# NOTE: not the perfect way of doing things, but it is a reasonable way for now.
def create_experiment_folder(project_folderpath, main_relfilepath,
        argnames, argvals_list, out_folderpath_argname, 
        exps_folderpath, readme, expname=None):
    
    assert folder_exists(exps_folderpath)
    assert expname is None or (not path_exists(
        join_paths([exps_folderpath, expname])))
    
    assert folder_exists(project_folderpath) and file_exists(join_paths([
        project_folderpath, main_relfilepath]))

    # create the main folder where things for the experiment will be.
    if expname is None:    
        expname = get_valid_name(exps_folderpath, "exp")
    ex_folderpath = join_paths([exps_folderpath, expname])
    create_folder(ex_folderpath)

    # write the config for the experiment
    write_jsonfile( 
        retrieve_values(locals(), [
            'project_folderpath', 'main_relfilepath',
            'argnames', 'argvals_list', 'out_folderpath_argname', 
            'exps_folderpath', 'readme', 'expname']),
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
        call_args = retrieve_values(locals(), [
            'project_folderpath', 'main_relfilepath', 'argnames', 'argvals'])
        script_filepath = join_paths([cfg_folderpath, 'run.sh'])
        create_run_script(out_filepath=script_filepath, **call_args)

        # write a config file for each configuration.
        write_jsonfile(
            create_dict(argnames, argvals), 
            join_paths([cfg_folderpath, 'config.json']))
    create_runall_script(ex_folderpath)

    return ex_folderpath

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

# perhaps can switch things around.


   

### TODO: upload to server should compress the files first and decompress them 
# there.

# may comment out if not needed.

# for boilerplate editing
def wrap_strings(s):
    return ', '.join(['\'' + a.strip() + '\'' for a in s.split(',')])


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

# 

# that can be part of the functionality of the research toolbox.

# as these 

# this model kind of suggests to included more information about the script 
# that we want to run and use it later.


# ./experiments/exp0/cfgs/cfg0/run.sh

## NOTE: can add things in such a way that it is possible to run it
# on the same folder. in the sense that it is going to change folders
# to the right place.

### and more.

# I think that I can get some of these variations by combining various iterators.

# a lot of experiments. I will try just to use some of the 
# experiments.

# TODO: 

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


### ignore_hidden_files and folders.

# perhaps for running, it is possible to get things to work.

# questions about the model. 

# there are options that upon copying may be set.

# generation of these results.

# I'm still deciding how capturing the output should be done.

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
gtx970_gpus = ["dual-970-0-%d" % i for i in xrange(13)] 
gtx980_gpus = ["quad-980-0-%d" % i for i in xrange(3)]
k40_gpus = ["quad-k40-0-0", "dual-k40-0-1", "dual-k40-0-2"]
titan_gpus = ["quad-titan-0-0"]
all_gpus = gtx970_gpus + gtx980_gpus + k40_gpus + titan_gpus

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