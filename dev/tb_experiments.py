
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
import research_toolbox.tb_experiments as tb_ex

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
    tb_ex.create_experiment_folder('beam_learn/main.py', argname_lst, argval_lst_lst,
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

    return tb_ex.map_experiment_folder(experiment_folderpath, _fn)

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

