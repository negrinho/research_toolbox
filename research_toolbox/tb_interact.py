import argparse
from collections import OrderedDict
import subprocess
import research_toolbox.tb_remote as tb_re
import research_toolbox.tb_resources as tb_rs
import research_toolbox.tb_logging as tb_lg
import research_toolbox.tb_io as tb_io
import research_toolbox.tb_filesystem as tb_fs

# check https://www.psc.edu/bridges/user-guide/running-jobs for more information
def write_server_run_script():
    assert servertype == 'bridges'
    # NOTE: to edit according to the configuration needed.
    jobname = 'benchmark'
    time_budget_in_hours = 48 # max 48 hours
    mem_budget_in_gb = 16
    partition_name = 'GPU-shared'
    num_cpus = 1 # probably ask a CPU for each GPU (or more if you have data loaders)
    num_gpus = 1 # up to 4 if k80, up to 2 if p100
    gpu_type = 'k80' # in ['k80', 'p100']

    script_header = [
        '#!/bin/bash',
        '#SBATCH --nodes=1',
        '#SBATCH --partition=%s' % partition_name,
        '#SBATCH --cpus-per-task=%d' % num_cpus,
        '#SBATCH --gres=gpu:%s:%d' % (gpu_type, num_gpus),
        '#SBATCH --mem=%dM' % tb_rs.convert_between_byte_units(mem_budget_in_gb,
            src_units='gigabytes', dst_units='megabytes'),
        '#SBATCH --time=%d' % tb_lg.convert_between_time_units(time_budget_in_hours,
            src_units='hours', dst_units='minutes'),
        '#SBATCH --job-name=%s' % jobname,
    ]
    # NOTE: changes to the environment can be put in the run script.
    script_body = [
        'module load tensorflow/1.5_gpu',
        'PYTHONPATH=%s:$PYTHONPATH' % remote_folderpath,
        'python %s > log_%s.txt' % (main_relfilepath, jobname)]

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
name_to_cmd['sync_and_run_on_server'] = lambda: (
    write_server_run_script(),
    name_to_cmd['sync_to_server'](),
    name_to_cmd['sync_from_server'](),
    name_to_cmd['run_on_server']())

if __name__ == '__main__':
    d = tb_io.read_jsonfile('./interact_config.json')

    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, choices=name_to_cmd.keys())
    parser.add_argument('server', type=str, choices=d['servers'].keys())
    parser.add_argument('job', type=str, choices=d['jobs'].keys())
    args = parser.parse_args()
    servertype = args.server

    local_folderpath = d['local_folderpath']
    servername = d['servers'][args.server]['servername']
    username = d['servers'][args.server]['username']
    remote_folderpath = d['servers'][args.server]['remote_folderpath']
    main_relfilepath = d['jobs'][args.job]['main_relfilepath']

    print name_to_cmd[args.cmd]()

# potentially add configs about the server more easily.
# it is possible to add a similar script for bridges as it is SLURM managed.
# check the run_on_matrix for more concrete details.