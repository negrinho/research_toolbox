import argparse
import itertools
import multiprocessing
import stat
import time
import psutil
import os
import research_toolbox.tb_filesystem as tb_fs
import research_toolbox.tb_io as tb_io
import research_toolbox.tb_logging as tb_lg
import research_toolbox.tb_utils as tb_ut
import research_toolbox.tb_random as tb_ra

class CommandLineArgs:
    def __init__(self, argname_prefix=''):
        self.parser = argparse.ArgumentParser()
        self.argname_prefix = argname_prefix

    def add(self, argname, argtype, default_value=None, optional=False, help=None,
            valid_value_lst=None, list_valued=False):
        valid_types = {'int' : int, 'str' : str, 'float' : float}
        assert argtype in valid_types

        nargs = None if not list_valued else '*'
        argtype = valid_types[argtype]

        self.parser.add_argument('--' + self.argname_prefix + argname,
            required=not optional, default=default_value, nargs=nargs,
            type=argtype, choices=valid_value_lst, help=help)

    def parse(self):
        return vars(self.parser.parse_args())

    def get_parser(self):
        return self.parser

def get_available_filename(folderpath, filename_prefix):
    idx = 0
    while True:
        name = "%s%d" % (filename_prefix, idx)
        path = tb_fs.join_paths([folderpath, name])
        if not tb_fs.path_exists(path):
            break
        else:
            idx += 1
    return name

# generating the call lines for a call to main.
def generate_call_lines(main_filepath,
        argname_lst, argvalue_lst,
        output_filepath=None, profile_filepath=None):

    sc_lines = ['python -u \\']
    # add the profiling instruction.
    if profile_filepath is not None:
        sc_lines += ['-m cProfile -o %s \\' % profile_filepath]
    sc_lines += ['%s \\' % main_filepath]
    # arguments for the call
    sc_lines += ['    --%s %s \\' % (k, v)
        for k, v in itertools.izip(argname_lst[:-1], argvalue_lst[:-1])]
    # add the output redirection.
    if output_filepath is not None:
        sc_lines += ['    --%s %s \\' % (argname_lst[-1], argvalue_lst[-1]),
                    '    > %s 2>&1' % (output_filepath)]
    else:
        sc_lines += ['    --%s %s' % (argname_lst[-1], argvalue_lst[-1])]
    return sc_lines

# all paths are relative to the current working directory or to entry folder path.
def create_run_script(main_filepath,
        argname_lst, argvalue_lst, script_filepath,
        # entry_folderpath=None,
        output_filepath=None, profile_filepath=None):

    sc_lines = [
        '#!/bin/bash',
        'set -e']
    # # change into the entry folder if provided.
    # if entry_folderpath is not None:
    #     sc_lines += ['cd %s' % entry_folderpath]
    # call the main function.
    sc_lines += generate_call_lines(**tb_ut.retrieve_values(locals(),
        ['main_filepath', 'argname_lst', 'argvalue_lst',
        'output_filepath', 'profile_filepath']))
    # change back to the previous folder if I change to some other folder.
    # if entry_folderpath is not None:
    #     sc_lines += ['cd -']
    tb_io.write_textfile(script_filepath, sc_lines, with_newline=True)
    # add run permissions.
    st = os.stat(script_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(script_filepath, st.st_mode | exec_bits)

# NOTE: can be done more concisely with a for loop.
def create_runall_script(experiment_folderpath):
    fo_names = tb_fs.list_folders(experiment_folderpath, recursive=False, use_relative_paths=True)
    num_exps = len([n for n in fo_names if tb_fs.path_last_element(n).startswith('cfg')])

    # creating the script.
    sc_lines = ['#!/bin/bash']
    sc_lines += [tb_fs.join_paths([experiment_folderpath, "cfg%d" % i, 'run.sh'])
        for i in xrange(num_exps)]

    # creating the run all script.
    out_filepath = tb_fs.join_paths([experiment_folderpath, 'run.sh'])
    tb_io.write_textfile(out_filepath, sc_lines, with_newline=True)
    st = os.stat(out_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(out_filepath, st.st_mode | exec_bits)

# NOTE: for now, this relies on the fact that upon completion of an experiment
# a results.json file, i.e., the existence of this file is used to determine
# if the folder related to this experiment has been run or not.
def create_runall_script_with_parallelization(experiment_folderpath):
    fo_names = tb_fs.list_folders(experiment_folderpath, recursive=False, use_relative_paths=True)
    num_exps = len([n for n in fo_names if tb_fs.path_last_element(n).startswith('cfg')])

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
        '        if [ ! -f %s ] || [ $force_rerun -eq 1 ]; then' % tb_fs.join_paths(
                    [experiment_folderpath, "cfg$i", 'results.json']),
        '            echo cfg$i',
        '            %s' % tb_fs.join_paths([experiment_folderpath, "cfg$i", 'run.sh']),
        '        fi',
        '    fi',
        '    i=$(($i + 1))',
        'done'
   ]
    # creating the run all script.
    out_filepath = tb_fs.join_paths([experiment_folderpath, 'run.sh'])
    tb_io.write_textfile(out_filepath, sc_lines, with_newline=True)
    st = os.stat(out_filepath)
    exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(out_filepath, st.st_mode | exec_bits)

# NOTE: not the perfect way of doing things, but it is a reasonable way for now.
# main_relfilepath is relative to the project folder path.
# entry_folderpath is the place it changes to before executing.
# if code and data folderpaths are provided, they are copied to the exp folder.
# all paths are relative I think that that is what makes most sense.
def create_experiment_folder(main_filepath,
        argname_lst, argval_lst_lst, output_folderpath_argname,
        all_experiments_folderpath, readme, experiment_name=None,
        # entry_folderpath=None,
        code_folderpath=None,
        # data_folderpath=None,
        capture_output=False, profile_run=False):

    assert tb_fs.folder_exists(all_experiments_folderpath)
    assert experiment_name is None or (not tb_fs.path_exists(
        tb_fs.join_paths([all_experiments_folderpath, experiment_name])))
    # assert folder_exists(project_folderpath) and file_exists(tb_fs.join_paths([
    #     project_folderpath, main_relfilepath]))

    # create the main folder where things for the experiment will be.
    if experiment_name is None:
        experiment_name = get_available_filename(all_experiments_folderpath, "exp")
    experiment_folderpath = tb_fs.join_paths([all_experiments_folderpath, experiment_name])
    tb_fs.create_folder(experiment_folderpath)

    # copy the code to the experiment folder.
    if code_folderpath is not None:
        code_foldername = tb_fs.path_last_element(code_folderpath)
        dst_code_fo = tb_fs.join_paths([experiment_folderpath, code_foldername])

        tb_fs.copy_folder(code_folderpath, dst_code_fo,
            ignore_hidden_files=True, ignore_hidden_folders=True,
            ignore_file_exts=['.pyc'])

        # change main_filepath to use that new code.
        main_filepath = tb_fs.join_paths([experiment_folderpath, main_filepath])

    # NOTE: no data copying for now because it often does not make much sense.
    data_folderpath = None ### TODO: remove later.
    # # copy the code to the experiment folder.
    # if data_folderpath is not None:
    #     data_foldername = path_last_element(data_folderpath)
    #     dst_data_fo = join_paths([experiment_folderpath, data_foldername])

    #     copy_folder(data_folderpath, dst_data_fo,
    #         ignore_hidden_files=True, ignore_hidden_folders=True)

    # write the config for the experiment.
    tb_io.write_jsonfile(
        tb_ut.retrieve_values(locals(), [
        'main_filepath',
        'argname_lst', 'argval_lst_lst', 'output_folderpath_argname',
        'all_experiments_folderpath', 'readme', 'experiment_name',
        'code_folderpath', 'data_folderpath',
        'capture_output', 'profile_run']),
        tb_fs.join_paths([experiment_folderpath, 'config.json']))

    # generate the executables for each configuration.
    argname_lst = list(argname_lst)
    argname_lst.append(output_folderpath_argname)
    for (i, vs) in enumerate(argval_lst_lst):
        cfg_folderpath = tb_fs.join_paths([experiment_folderpath, "cfg%d" % i])
        tb_fs.create_folder(cfg_folderpath)

        # create the script
        argvalue_lst = list(vs)
        argvalue_lst.append(cfg_folderpath)
        call_args = tb_ut.retrieve_values(locals(),
            ['argname_lst', 'argvalue_lst', 'main_filepath'])

        call_args['script_filepath'] = tb_fs.join_paths([cfg_folderpath, 'run.sh'])
        if capture_output:
            call_args['output_filepath'] = tb_fs.join_paths(
                [cfg_folderpath, 'output.txt'])
        if profile_run:
            call_args['profile_filepath'] = tb_fs.join_paths(
                [cfg_folderpath, 'profile.txt'])
        create_run_script(**call_args)

        # write a config file for each configuration
        tb_io.write_jsonfile(
            tb_ut.create_dict(argname_lst, argvalue_lst),
            tb_fs.join_paths([cfg_folderpath, 'config.json']))
    # create_runall_script(experiment_folderpath)
    create_runall_script_with_parallelization(experiment_folderpath)

    return experiment_folderpath

### tools for processing the experiments folders once they have finished.
def map_experiment_folder(experiment_folderpath, fn):
    fo_paths = tb_fs.list_folders(experiment_folderpath, recursive=False, use_relative_paths=False)
    num_exps = len([p for p in fo_paths if tb_fs.path_last_element(p).startswith('cfg')])

    ps = []
    rs = []
    for i in xrange(num_exps):
        p = tb_fs.join_paths([experiment_folderpath, 'cfg%d' % i])
        rs.append(fn(p))
        ps.append(p)
    return (ps, rs)

def load_experiment_folder(experiment_folderpath, json_filename_lst,
    abort_if_notexists=True, only_load_if_all_exist=False):
    def _fn(cfg_path):
        ds = []
        for name in json_filename_lst:
            p = tb_fs.join_paths([cfg_path, name])

            if (not abort_if_notexists) and (not tb_fs.file_exists(p)):
                d = None
            # if abort if it does not exist, it is going to fail reading the file.
            else:
                d = tb_io.read_jsonfile(p)
            ds.append(d)
        return ds

    (ps, rs) = map_experiment_folder(experiment_folderpath, _fn)

    # filter only the ones that loaded successfully all files.
    if only_load_if_all_exist:
        proc_ps = []
        proc_rs = []
        for i in xrange(len(ps)):
            if all([x is not None for x in rs[i]]):
                proc_ps.append(ps[i])
                proc_rs.append(rs[i])
        (ps, rs) = (proc_ps, proc_rs)

    return (ps, rs)

def generate_config_args(d, ortho=False):
    ks = d.keys()
    if not ortho:
        vs_list = tb_ut.iter_product([d[k] for k in ks])
    else:
        vs_list = tb_ut.iter_ortho_all([d[k] for k in ks], [0] * len(ks))

    argval_lst_lst = []
    for vs in vs_list:
        proc_v = []
        for k, v in itertools.izip(ks, vs):
            if isinstance(k, tuple):
                # if it is iterable, unpack v
                if isinstance(v, list) or isinstance(v, tuple):
                    assert len(k) == len(v)
                    proc_v.extend(v)
                # if it is not iterable, repeat a number of times equal
                # to the size of the key.
                else:
                    proc_v.extend([v] * len(k))
            else:
                proc_v.append(v)
        argval_lst_lst.append(proc_v)

    # unpacking if there are multiple tied argname_lst
    argname_lst = []
    for k in ks:
        if isinstance(k, tuple):
            argname_lst.extend(k)
        else:
            argname_lst.append(k)

    # guarantee no repeats.
    assert len(set(argname_lst)) == len(argname_lst)

    # resorting the tuples according to sorting permutation.
    idxs = tb_ra.argsort(argname_lst, [lambda x: x])
    argname_lst = tb_ra.apply_permutation(argname_lst, idxs)
    argval_lst_lst = [tb_ra.apply_permutation(vs, idxs) for vs in argval_lst_lst]
    return (argname_lst, argval_lst_lst)

# NOTE: this has been made a bit restrictive, but captures the main functionality
# that it is required to generate the experiments.
def copy_regroup_config_generator(d_gen, d_update):
    # all keys in the regrouping dictionary have to be in the original dict
    flat_ks = []
    for k in d_update:
        if isinstance(k, tuple):
            assert all([ki in d_gen for ki in k])
            flat_ks.extend(k)
        else:
            assert k in d_gen
            flat_ks.append(k)

    # no tuple keys. NOTE: this can be relaxed by flattening, and reassigning
    # but this is more work.
    assert all([not isinstance(k, tuple) for k in d_gen])
    # no keys that belong to multiple groups.
    assert len(flat_ks) == len(set(flat_ks))

    # regrouping of the dictionary.
    proc_d = dict(d_gen)
    for (k, v) in d_update.iteritems():
        # check that the dimensions are consistent.
        assert all([
            ((not isinstance(vi, tuple)) and (not isinstance(vi, tuple))) or
            len(vi) == len(k)
                for vi in v])

        if isinstance(k, tuple):
            # remove the original keys
            map(proc_d.pop, k)
            proc_d[k] = v
        else:
            proc_d[k] = v

    return proc_d

# TODO: perhaps fix the API with regards to kwargs for consistency with
# other examples.
def run_guarded_experiment(experiment_fn, maxmemory_mbs, maxtime_secs, **kwargs):
        start = time.time()

        p = multiprocessing.Process(target=experiment_fn, kwargs=kwargs)
        p.start()
        while p.is_alive():
            p.join(1.0)
            try:
                mbs_p = tb_lg.memory_process(p.pid)
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

class ArgsDict:
    def __init__(self, filepath=None):
        self.d = {}
        if filepath != None:
            self._read(filepath)

    def set_arg(self, key, val, abort_if_exists=True):
        assert (not abort_if_exists) or key not in self.d
        assert (type(key) == str and len(key) > 0)
        self.d[key] = val

    def write(self, filepath):
        tb_io.write_jsonfile(self.d, filepath)

    def _read(self, filepath):
        return tb_io.read_jsonfile(filepath)

    def get_dict(self):
        return dict(self.d)

class SummaryDict:
    def __init__(self, filepath=None, abort_if_different_lengths=False):
        self.abort_if_different_lengths = abort_if_different_lengths
        if filepath != None:
            self.d = self._read(filepath)
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

    # NOTE: maybe write is not useful
    def write(self, filepath):
        tb_io.write_jsonfile(self.d, filepath)

    def _read(self, filepath):
        return tb_io.read_jsonfile(filepath)

    def _check_consistency(self):
        assert (not self.abort_if_different_lengths) or (len(
            set([len(v) for v in self.d.itervalues()])) <= 1)

    def get_dict(self):
        return dict(self.d)

##### TODO: all this is very preliminary. (edit or remove in future commits.)
### NOTE: below is an example set of functionalities convenient to create
# extensive experiments using whatever function we have at hand.
# NOTE: these may require some adaptation to be practical in a new example.
def create_experimsent_from_fn(fn, overwrite=False):
    experiment_name = fn.__name__
    experiment_folderpath = tb_fs.join_paths(['experiments', experiment_name])
    if overwrite and tb_fs.folder_exists(experiment_folderpath):
        tb_fs.delete_folder(experiment_folderpath, False)

    cfgs = fn()
    argname_lst = cfgs[0].keys()
    argval_lst_lst = [[d[k] for k in argname_lst] for d in cfgs]
    create_experiment_folder('beam_learn/main.py', argname_lst, argval_lst_lst,
        'out_folder', 'experiments', '', experiment_name, 'beam_learn', True)


import pandas as pd

pd.set_option('display.max_colwidth', -1)

# NOTE: this is an example that was useful for particular experiment and
# that now can be adapted for new cases.
def explore_experiment(experiment_folderpath, use_checkpoints=False):
    def _fn(e_folderpath):
        cfg = tb_io.read_jsonfile(tb_fs.join_paths([e_folderpath, 'config.json']))

        res = None
        if not use_checkpoints:
            res_fpath = tb_fs.join_paths([e_folderpath, 'results.json'])
        else:
            res_fpath = tb_fs.join_paths([e_folderpath, 'checkpoint.json'])

        if tb_fs.file_exists(res_fpath):
            res = tb_io.read_jsonfile(res_fpath)
        return (cfg, res)

    return map_experiment_folder(experiment_folderpath, _fn)

def load_all_experiments(load_unfinished=False):
    xs = [explore_experiment(p)
        for p in tb_fs.list_folders('experiments')]
    return xs
import numpy as np
from pprint import pprint, pformat
import subprocess

# TODO: this one needs to be adapted.
def summarize_results(d):
    i = np.argmax(d['dev_accuracy'])

    # additional information for inspection.
    d['dev_accuracy_best_epoch'] = i
    d['dev_accuracy_at_20'] = np.max(d['dev_accuracy'][:20])
    d['cfg_name'] = tb_fs.path_last_element(d['out_folder'])
    d['cfg_id'] = int(d['cfg_name'].lstrip('cfg'))

    d['dev_accuracy'] = d['dev_accuracy'][i]
    d['test_accuracy'] = d['test_accuracy'][i]
    d['epoch_mins'] = np.mean(d['epoch_mins'])
    d['epoch_mbs'] = np.max(d['epoch_mbs'])
    del d['used_lr']

    return d

def keys_with_variation(ds):
    return tb_ut.filter_dict(tb_ut.key_to_values(ds), lambda k, v: len(v) > 1)

def get_float_formatter(num_decimals, multiplier=1.0):
    def fn(x):
        return unicode('{0:.{1}f}'.format(
            np.round(multiplier * x, num_decimals), num_decimals))
    return fn

def create_table_from_experiment(experiment_name, rows, columns, values,
        abort_if_incomplete_configs=True, use_checkpoints=False,
        single_row_multitable=False, print_to_terminal=True,
        max_column_width=10 ** 9, abort_if_different_keys=True):

    _, xs = explore_experiment('experiments/%s' % experiment_name, use_checkpoints)

    cfgs = []
    res = []
    for (c, r) in xs:
        if r is not None:
            cfgs.append(c)
            res.append(r)
        else:
            assert not abort_if_incomplete_configs
    xs = tb_ut.zip_toggle([cfgs, res])

    ks = keys_with_variation(cfgs)
    c = dict(cfgs[0])
    for k in ks:
        c.pop(k)

    ks.pop('out_folder')
    print "***%s***" % experiment_name
    pprint(ks)
    print

    ds = [summarize_results(tb_ut.merge_dicts(x)) for x in xs]

    # if the values are with respective
    if any([v in values for v in [
        'dev_precision', 'dev_recall', 'dev_fb1',
        'test_precision', 'test_recall', 'test_fb1']]):

        def _extract_fn(fpath):

            out = subprocess.check_output(
                ["cat %s | data/conll_2000/conlleval.txt" % fpath], shell=True)

            res_line = out.split('\n')[1]
            f1 = float(res_line.split(';')[-1].split(": ")[1])

            p, r, fb1 = map(lambda x: 0.01 * float(x.split(': ')[1]) ,
                res_line.split('%; '))[1:]

            return p, r, fb1

        # add the test and dev performances to the file.
        for d in ds:
            (d['dev_precision'], d['dev_recall'], d['dev_fb1']) = _extract_fn(
                tb_fs.join_paths([d['out_folder'], 'pred_dev.txt']))

            (d['test_precision'], d['test_recall'], d['test_fb1']) = _extract_fn(
                tb_fs.join_paths([d['out_folder'], 'pred_test.txt']))

            # this is the final, last run for conll2000
            fpath = tb_fs.join_paths([d['out_folder'], 'final_pred_test.txt'])
            if tb_fs.file_exists(fpath):

                (d['final_test_precision'],
                    d['final_test_recall'],
                    d['final_test_fb1']) = _extract_fn(fpath)

    df = tb_ut.create_dataframe(ds, abort_if_different_keys)

    # # shorten the names appropriately.
    df = df.rename(columns={k : k[:max_column_width] for k in rows})
    rows = [k[:max_column_width] for k in rows]

    # determines teh table layout.
    if not single_row_multitable:

        ts = [df.pivot_table(
            index=rows,
            columns=columns,
            values=[v]) for v in values]

    else:
        ts = [df.pivot_table(index=rows, columns=columns, values=values
        )#.sort_values('dev_accuracy', ascending=False)
        ]

    tb_fs.create_folder('analyses/%s' % experiment_name, abort_if_exists=False)
    s_c = pformat(c)
    ss_df = [t.to_string(float_format=get_float_formatter(2, 100.0)) for t in ts]

    lines = [s_c]
    for s in ss_df:
        lines.append('')
        lines.append(s)

    if print_to_terminal:
        # print to terminal
        for s in lines:
            print s

    # write to file
    tb_io.write_textfile('analyses/%s/results.txt' % experiment_name, lines)
    tb_io.write_csvfile(ds, 'analyses/%s/results.csv' % experiment_name,
        sort_keys=True, abort_if_different_keys=abort_if_different_keys)

def create_table(experiment_name, bio=False, use_checkpoints=False):
    print experiment_name
    vals = ['dev_accuracy', 'test_accuracy']
    if bio:
        vals += ['test_fb1', 'test_precision', 'test_recall']

    create_table_from_experiment(experiment_name,
        ['data_type', 'alias'], [],
        # ['data_type', 'train_type'],
        # ['loss_type', 'train_beam_size'],
        vals,
        abort_if_incomplete_configs=False, use_checkpoints=use_checkpoints,
        single_row_multitable=True, abort_if_different_keys=False)

