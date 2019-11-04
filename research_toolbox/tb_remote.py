### running remotely
import subprocess
import paramiko
import getpass
import inspect
import uuid
import urllib.request, urllib.parse, urllib.error
import research_toolbox.tb_utils as tb_ut
import research_toolbox.tb_resources as tb_rs
import research_toolbox.tb_logging as tb_lg
import research_toolbox.tb_filesystem as tb_fs


def get_password():
    return getpass.getpass()


def run_on_server(bash_command,
                  servername,
                  username=None,
                  password=None,
                  folderpath=None,
                  wait_for_output=True,
                  prompt_for_password=False):
    """SSH into a machine and runs a bash command there. Can specify a folder
    to change directory into after logging in.
    """

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
    _, stdout, stderr = sess.exec_command(bash_command)

    # depending if waiting for output or not.
    if wait_for_output:
        stdout_lines = stdout.readlines()
        stderr_lines = stderr.readlines()
    else:
        stdout_lines = None
        stderr_lines = None

    sess.close()
    return stdout_lines, stderr_lines


# tools for handling the lithium server, which has no scheduler.
def get_lithium_nodes():
    return {
        'gtx970': ["dual-970-0-%d" % i for i in range(0, 13)],
        'gtx980': ["quad-980-0-%d" % i for i in [0, 1, 2]],
        'k40': ["quad-k40-0-0", "dual-k40-0-1", "dual-k40-0-2"],
        'titan': ["quad-titan-0-0"]
    }


def get_lithium_resource_availability(servername,
                                      username,
                                      password=None,
                                      abort_if_any_node_unavailable=True,
                                      nodes_to_query=None):

    # prompting for password if asked about. (because lithium needs password)
    if password == None:
        password = getpass.getpass()

    # query all nodes if None
    if nodes_to_query == None:
        nodes_to_query = tb_ut.flatten(get_lithium_nodes())

    # script to define the functions to get the available resources.
    cmd_lines = ['import psutil', 'import subprocess', 'import numpy as np', '']
    fns = [
        tb_rs.convert_between_byte_units, tb_rs.cpus_total, tb_rs.memory_total,
        tb_rs.gpus_total, tb_rs.cpus_free, tb_rs.memory_free, tb_rs.gpus_free,
        tb_rs.gpus_free_ids
    ]

    for fn in fns:
        cmd_lines += [
            line.rstrip('\n') for line in inspect.getsourcelines(fn)[0]
        ]
        cmd_lines.append('')

    cmd_lines += [
        'print \'%d;%.2f;%d;%d;%.2f;%d;%s\' % ('
        'cpus_total(), memory_total(), gpus_total(), '
        'cpus_free(), memory_free(), gpus_free(), '
        '\' \'.join(map(str, gpus_free_ids())))'
    ]
    py_cmd = "\n".join(cmd_lines)

    ks = [
        'cpus_total', 'mem_mbs_total', 'gpus_total', 'cpus_free',
        'mem_mbs_free', 'gpus_free', 'free_gpu_ids'
    ]

    # run the command to query the information.
    write_script_cmd = 'echo \"%s\" > avail_resources.py' % py_cmd
    run_on_server(write_script_cmd, servername, username, password)

    # get it for each of the models
    node_to_resources = {}
    for host in nodes_to_query:
        cmd = 'ssh -T %s python avail_resources.py' % host
        stdout_lines, stderr_lines = run_on_server(cmd, servername, username,
                                                   password)
        # print stdout_lines, stderr_lines

        # if it did not fail.
        if len(stdout_lines) == 1:
            # extract the actual values from the command line.
            str_vs = stdout_lines[0].strip().split(';')
            assert len(str_vs) == 7
            # print str_vs
            vs = [
                fn(str_vs[i])
                for (i, fn) in enumerate([int, float, int] * 2 + [str])
            ]
            vs[-1] = [int(x) for x in vs[-1].split(' ') if x != '']

            d = tb_ut.create_dict(ks, vs)
            node_to_resources[host] = d
        else:
            assert not abort_if_any_node_unavailable
            node_to_resources[host] = None

    delete_script_cmd = 'rm avail_resources.py'
    run_on_server(delete_script_cmd, servername, username, password)
    return node_to_resources


# TODO: add functionality to check if the visible gpus are busy or not
# and maybe terminate upon that event.
# running on one of the compute nodes.
# NOTE: this function has minimum error checking.
def run_on_lithium_node(bash_command,
                        node,
                        servername,
                        username,
                        password=None,
                        visible_gpu_ids=None,
                        folderpath=None,
                        wait_for_output=True,
                        run_on_head_node=False):
    # check that node exists.
    assert node in tb_ut.flatten(get_lithium_nodes())

    # prompting for password if asked about. (because lithium needs password)
    if password == None:
        password = getpass.getpass()

    # if no visilbe gpu are specified, it creates a list with nothing there.
    if visible_gpu_ids is None:
        visible_gpu_ids = []

    # creating the command to run remotely.
    gpu_cmd = 'export CUDA_VISIBLE_DEVICES=%s' % ",".join(
        map(str, visible_gpu_ids))
    if not run_on_head_node:
        cmd = "ssh -T %s \'%s && %s\'" % (node, gpu_cmd, bash_command)
    else:
        # NOTE: perhaps repetition could be improved here. also, probably head
        # node does not have gpus.
        cmd = "%s && %s" % (gpu_cmd, bash_command)

    return run_on_server(
        cmd,
        **tb_ut.subset_dict_via_selection(locals(), [
            'servername', 'username', 'password', 'folderpath',
            'wait_for_output'
        ]))


# TODO: add functionality to run on all lithium node.
# NOTE: this may require adding some information to the server.
# NOTE: if any of the command waits for output, it will mean that
# it is going to wait until completion of that command until doing the other one.
# NOTE: as lithium does not have a scheduler, resource management has to be done
# manually. This one has to prompt twice.
# NOTE: the time budget right now does not do anything.
# TODO: run on head node should not be true for these, because I
# do all the headnode information here.
class LithiumRunner:

    def __init__(self,
                 servername,
                 username,
                 password=None,
                 only_run_if_can_run_all=True):
        self.servername = servername
        self.username = username
        self.password = password if password is not None else get_password()
        self.jobs = []

    def register(self,
                 bash_command,
                 num_cpus=1,
                 num_gpus=0,
                 mem_budget=8.0,
                 time_budget=60.0,
                 mem_units='gigabytes',
                 time_units='minutes',
                 folderpath=None,
                 wait_for_output=True,
                 require_gpu_types=None,
                 require_nodes=None,
                 run_on_head_node=False):

        # NOTE: this is not implemented for now.
        assert not run_on_head_node
        # should not specify both.
        assert require_gpu_types is None or require_nodes is None

        self.jobs.append(
            tb_ut.subset_dict_via_selection(locals(), [
                'bash_command', 'num_cpus', 'num_gpus', 'mem_budget',
                'time_budget', 'mem_units', 'time_units', 'folderpath',
                'wait_for_output', 'require_gpu_types', 'require_nodes',
                'run_on_head_node'
            ]))

    def run(self, run_only_if_enough_resources_for_all=True):
        args = tb_ut.subset_dict_via_selection(
            vars(self), ['servername', 'username', 'password'])
        args['abort_if_any_node_unavailable'] = False

        # get the resource availability and filter out unavailable nodes.
        d = get_lithium_resource_availability(**args)
        d = {k: v for (k, v) in d.items() if v is not None}

        g = get_lithium_nodes()

        # assignments to each of the registered jobs
        run_cfgs = []
        for x in self.jobs:
            if x['require_nodes'] is not None:
                req_nodes = x['require_nodes']
            else:
                req_nodes = list(d.keys())

            # based on the gpu type restriction.
            if x['require_gpu_types'] is not None:
                req_gpu_nodes = tb_ut.flatten(
                    tb_ut.subset_dict_via_selection(g, x['require_gpu_types']))
            else:
                # NOTE: only consider the nodes that are available anyway.
                req_gpu_nodes = list(d.keys())

            # potentially available nodes to place this job.
            nodes = list(set(req_nodes).intersection(req_gpu_nodes))
            assert len(nodes) > 0

            # greedy assigned to a node.
            assigned = False
            for n in nodes:
                r = d[n]
                # if there are enough resources on the node, assign it to the
                # job.
                if ((r['cpus_free'] >= x['num_cpus']) and
                    (r['gpus_free'] >= x['num_gpus']) and
                    (r['mem_mbs_free'] >= tb_rs.convert_between_byte_units(
                        x['mem_budget'],
                        src_units=x['mem_units'],
                        dst_units='megabytes'))):

                    # record information about where to run the job.
                    run_cfgs.append({
                        'node':
                        n,
                        'visible_gpu_ids':
                        r['free_gpu_ids'][:x['num_gpus']]
                    })

                    # deduct the allocated resources from the available resources
                    # for that node.
                    r['cpus_free'] -= x['num_cpus']
                    r['gpus_free'] -= x['num_gpus']
                    r['mem_mbs_free'] -= tb_rs.convert_between_byte_units(
                        x['mem_budget'],
                        src_units=x['mem_units'],
                        dst_units='megabytes')
                    r['free_gpu_ids'] = r['free_gpu_ids'][x['num_gpus']:]
                    # assigned = True
                    break

            # if not assigned, terminate without doing anything.
            if not assigned:
                run_cfgs.append(None)
                if run_only_if_enough_resources_for_all:
                    print(("Insufficient resources to satisfy"
                          " (cpus=%d, gpus=%d, mem=%0.3f%s)" %
                          (x['num_cpus'], x['num_gpus'], x['mem_budget'],
                           x['mem_units'])))
                    return None

        # running the jobs that have a valid config.
        remaining_jobs = []
        outs = []
        for x, c in zip(self.jobs, run_cfgs):
            if c is None:
                remaining_jobs.append(x)
            else:
                out = run_on_lithium_node(**tb_ut.merge_dicts([
                    tb_ut.subset_dict_via_selection(
                        vars(self), ['servername', 'username', 'password']),
                    tb_ut.subset_dict_via_selection(x, [
                        'bash_command', 'folderpath', 'wait_for_output',
                        'run_on_head_node'
                    ]),
                    tb_ut.subset_dict_via_selection(c,
                                                    ['node', 'visible_gpu_ids'])
                ]))
                outs.append(out)

        self.jobs = remaining_jobs
        return outs


# NOTE: there may exist problems due to race conditions; can be solved later.
def run_on_matrix(bash_command,
                  servername,
                  username,
                  password=None,
                  num_cpus=1,
                  num_gpus=0,
                  mem_budget=8.0,
                  time_budget=60.0,
                  mem_units='gigabytes',
                  time_units='minutes',
                  folderpath=None,
                  wait_for_output=True,
                  require_gpu_type=None,
                  run_on_head_node=False,
                  jobname=None):

    assert (not run_on_head_node) or num_gpus == 0
    assert require_gpu_type is None  ### NOT IMPLEMENTED YET.

    # prompts for password if it has not been provided
    if password == None:
        password = getpass.getpass()

    script_cmd = "\n".join(['#!/bin/bash', bash_command])
    script_name = "run_%s.sh" % uuid.uuid4()

    # either do the call using sbatch, or run directly on the head node.
    if not run_on_head_node:
        cmd_parts = [
            'srun' if wait_for_output else 'sbatch',
            '--cpus-per-task=%d' % num_cpus,
            '--gres=gpu:%d' % num_gpus,
            '--mem=%d' % tb_rs.convert_between_byte_units(
                mem_budget, src_units=mem_units, dst_units='megabytes'),
            '--time=%d' % tb_lg.convert_between_time_units(
                time_budget, time_units, dst_units='minutes')
        ]
        if jobname is not None:
            cmd_parts += ['--job-name=%s' % jobname]
        cmd_parts += [script_name]

        run_script_cmd = ' '.join(cmd_parts)
    else:
        run_script_cmd = './' + script_name

    # actual command to run remotely
    remote_cmd = " && ".join([
        "echo \'%s\' > %s" % (script_cmd, script_name),
        "chmod +x %s" % script_name, run_script_cmd,
        "rm %s" % script_name
    ])

    return run_on_server(
        remote_cmd,
        **tb_ut.subset_dict_via_selection(locals(), [
            'servername', 'username', 'password', 'folderpath',
            'wait_for_output'
        ]))


def rsync_options(recursive=True,
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


# NOTE: that because this is rsync, you have to be careful about
# the last /. perhaps add a condition that makes this easier to handle.
def sync_local_folder_from_local(src_folderpath,
                                 dst_folderpath,
                                 only_transfer_newer_files=True):

    cmd = ['rsync', '--verbose', '--archive']
    # whether to only transfer newer files or not.
    if only_transfer_newer_files:
        cmd += ['--update']

    # add the backslash if it does not exist. for working correctly with rsync.
    if src_folderpath[-1] != '/':
        src_folderpath = src_folderpath + '/'
    if dst_folderpath[-1] != '/':
        dst_folderpath = dst_folderpath + '/'

    cmd += [src_folderpath, dst_folderpath]
    out = subprocess.check_output(cmd)

    return out


# NOTE: will always prompt for password due to being the simplest approach.
# if a password is necessary, it will prompt for it.
# dst_folderpath should already be created.
def sync_remote_folder_from_local(src_folderpath,
                                  dst_folderpath,
                                  servername,
                                  username=None,
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


def sync_local_folder_from_remote(src_folderpath,
                                  dst_folderpath,
                                  servername,
                                  username=None,
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


def download_file(urlpath, folderpath, filename=None,
                  abort_if_file_exists=True):
    if filename is None:
        filename = urlpath.split('/')[-1]
    filepath = tb_fs.join_paths([folderpath, filename])
    assert tb_fs.folder_exists(folderpath)
    assert (not tb_fs.file_exists(filepath)) or abort_if_file_exists
    f = urllib.request.URLopener()
    f.retrieve(urlpath, filepath)