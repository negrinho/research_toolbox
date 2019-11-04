import argparse
from collections import OrderedDict
import subprocess
import research_toolbox.tb_remote as tb_re
import research_toolbox.tb_resources as tb_rs
import research_toolbox.tb_logging as tb_lg
import research_toolbox.tb_io as tb_io
import research_toolbox.tb_filesystem as tb_fs
import research_toolbox.tb_experiments as tb_ex

import importlib
from pprint import pprint


class CommandLineDynamicParser:

    def __init__(self, max_num_args):
        self.max_num_args = max_num_args
        self.parser = tb_ex.CommandLineArgs()
        for i in range(max_num_args):
            self.parser.add('arg%d_name' % i, 'str', optional=True)
            self.parser.add('arg%d_val' % i, 'str', optional=True)
            self.parser.add(
                'arg%d_type' % i,
                'str',
                valid_value_lst=[
                    'int', 'float', 'str', 'int_list', 'float_list', 'str_list'
                ],
                optional=True)

    def parse(self):
        d = self.parser.parse()
        for i in range(self.max_num_args):
            arg_ks = ['arg%d_name' % i, 'arg%d_val' % i, 'arg%d_type' % i]
            assert all(k is None for k in arg_ks) or all(
                k is not None for k in arg_ks)
            if all(d[k] is None for k in arg_ks):
                for k in arg_ks:
                    d.pop(k)

        ks = set([x.split('_')[0] for x in list(d.keys())])
        arg_type_to_fn = {
            'int': int,
            'str': str,
            'float': float,
            'int_list': lambda x: list(map(int, x.split())),
            'str_list': lambda x: list(map(str, x.split())),
            'float_list': lambda x: list(map(float, x.split())),
        }

        proc_d = {}
        for k in ks:
            arg_name = d[k + '_name']
            arg_val = d[k + '_val']
            arg_type = d[k + '_type']
            proc_d[arg_name] = arg_type_to_fn[arg_type](arg_val)

        return proc_d

    def get_parser(self):
        return self.parser.get_parser()


def run_function(import_name,
                 function_name,
                 argname_to_argval,
                 print_results=True):
    module = importlib.import_module(import_name)
    r = getattr(module, function_name)(**argname_to_argval)
    if print_results:
        pprint(r)


# check https://www.psc.edu/bridges/user-guide/running-jobs for more information
def write_server_run_script():
    assert servertype == 'bridges'
    # NOTE: to edit according to the configuration needed.
    jobname = jobtype
    time_budget_in_hours = 48  # max 48 hours
    mem_budget_in_gb = 16
    partition_name = 'GPU-shared'
    num_cpus = 1  # probably ask a CPU for each GPU (or more if you have data loaders)
    num_gpus = 1  # up to 4 if k80, up to 2 if p100
    gpu_type = 'k80'  # in ['k80', 'p100']

    script_header = [
        '#!/bin/bash',
        '#SBATCH --nodes=1',
        '#SBATCH --partition=%s' % partition_name,
        '#SBATCH --cpus-per-task=%d' % num_cpus,
        '#SBATCH --gres=gpu:%s:%d' % (gpu_type, num_gpus),
        '#SBATCH --mem=%dM' % tb_rs.convert_between_byte_units(
            mem_budget_in_gb, src_units='gigabytes', dst_units='megabytes'),
        '#SBATCH --time=%d' % tb_lg.convert_between_time_units(
            time_budget_in_hours, src_units='hours', dst_units='minutes'),
        '#SBATCH --job-name=%s' % jobname,
    ]
    # NOTE: changes to the environment can be put in the run script.
    script_body = [
        'module load tensorflow/1.5_gpu',
        'PYTHONPATH=%s:$PYTHONPATH' % remote_folderpath,
        'python -u %s > log_%s.txt' % (main_relfilepath, jobname)
    ]

    script_filepath = tb_fs.join_paths([local_folderpath, "run.sh"])
    tb_io.write_textfile(script_filepath, script_header + [''] + script_body)
    subprocess.check_output(['chmod', '+x', script_filepath])


name_to_cmd = OrderedDict()
name_to_cmd['sync_from_server'] = lambda: tb_re.sync_local_folder_from_remote(
    src_folderpath=remote_folderpath,
    dst_folderpath=local_folderpath,
    servername=servername,
    username=username,
    only_transfer_newer_files=True)
name_to_cmd['sync_to_server'] = lambda: tb_re.sync_remote_folder_from_local(
    src_folderpath=local_folderpath,
    dst_folderpath=remote_folderpath,
    servername=servername,
    username=username,
    only_transfer_newer_files=True)
name_to_cmd['run_on_server'] = lambda: tb_re.run_on_server(
    bash_command='sbatch run.sh',
    servername=servername,
    username=username,
    folderpath=remote_folderpath,
    wait_for_output=True,
    prompt_for_password=False)
name_to_cmd['sync_and_run_on_server'] = lambda: (write_server_run_script(
), name_to_cmd['sync_to_server'](), name_to_cmd['sync_from_server'](),
                                                 name_to_cmd['run_on_server']())

if __name__ == '__main__':
    d = tb_io.read_jsonfile('./interact_config.json')

    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, choices=list(name_to_cmd.keys()))
    parser.add_argument('server', type=str, choices=list(d['servers'].keys()))
    parser.add_argument('job', type=str, choices=list(d['jobs'].keys()))
    args = parser.parse_args()
    servertype = args.server
    jobtype = args.job

    local_folderpath = d['local_folderpath']
    servername = d['servers'][args.server]['servername']
    username = d['servers'][args.server]['username']
    remote_folderpath = d['servers'][args.server]['remote_folderpath']
    main_relfilepath = d['jobs'][args.job]['main_relfilepath']

    print(name_to_cmd[args.cmd]())

# potentially add configs about the server more easily.
# it is possible to add a similar script for bridges as it is SLURM managed.
# check the run_on_matrix for more concrete details.