import subprocess
import paramiko
import getpass
import inspect
import uuid
import tb_utils as tb_ut
import tb_resources as tb_rs
import tb_logging as tb_lg


def connect(max_password_attempts,password,prompt_for_password,servername,username):
    # if everything works return a session
    # This assumes that if the password is passed in as an argument, it shall be
    # correct

    sess = paramiko.SSHClient()
    sess.load_system_host_keys()
    #ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    
    sess.set_missing_host_key_policy(paramiko.AutoAddPolicy())


    for i in range(max_password_attempts):
        if password != None:
            sess.connect(servername, username = username, password=password)
            break

        if password == None and prompt_for_password:
            password = getpass.getpass(prompt = username+"@"+servername+"'s password: ")

        try:
            sess.connect(servername, username=username, password=password)
            break

        except paramiko.AuthenticationException:
            if i == max_password_attempts - 1:
                # We have failed too many times. Return directly
                print "Login max attempts limit reached. Terminating..."
                return None

            password = None
            print "Access denied. Please re-enter the password."

    assert sess != None # ensures the session is successfully established
    print "Successfully logged in."
    return sess


def run_on_server(bash_command, servername, username=None, password=None,
        folderpath=None, wait_for_output=True, prompt_for_password=False, 
        max_password_attempts = 3):

    """
    SSH into a machine and runs a bash command there. Can specify a folder 
    to change directory into after logging in.
    """


    # getting credentials
    if username != None: 
        host = username + "@" + servername 
    else:
        host = servername
    
    

    if not wait_for_output:
        bash_command = "nohup %s &" % bash_command

    if folderpath != None:
        bash_command = "cd %s && %s" % (folderpath, bash_command) 

    
    sess = connect(max_password_attempts,password,prompt_for_password,servername,username)

    
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

# bash_command = "python playground.py"
# servername = "bridges.psc.xsede.org"
# username = "zejiea"
# folderpath = "~/misc"
# print run_on_server(bash_command,servername,username=username,prompt_for_password=True ,folderpath = folderpath)


def run_on_bridges(bash_command, username, servername = 'bridges.psc.xsede.org', password = None,
        num_cpus=1, num_gpus=0, num_tasks=1,
        mem_budget= 4.0, time_budget = 60.0, mem_units='gb', time_units='m',
        folderpath= None, wait_for_output= True,
        require_gpu_type= 'k80', run_on_head_node = False, jobname=None,gpu_node_type = 'GPU-shared'):

    assert (not run_on_head_node) or num_gpus == 0

    if password == None:
        prompt_for_password = True
    else:
        prompt_for_password = False
    script_cmd = "\n".join(['#!/bin/bash', bash_command])
    script_name = "run_%s.sh" % uuid.uuid4()

    # either do the call using sbatch, or run directly on the head node.
    if not run_on_head_node:
        cmd_parts = ['srun' if wait_for_output else 'sbatch',
            '--cpus-per-task=%d' % num_cpus,
            '-n %d' % num_tasks,
            '--mem=%d' % tb_rs.convert_between_byte_units(mem_budget,
                src_units=mem_units, dst_units='mb'),
            '--time=%d' % tb_lg.convert_between_time_units(time_budget,
                time_units, dst_units='m'),]

        if num_gpus > 0:
            cmd_parts += ['--gres=gpu:%s:%d' % (require_gpu_type,num_gpus),'-p %s' % gpu_node_type]

        if jobname is not None:
            cmd_parts += ['--job-name=%s' % jobname]

        cmd_parts += [script_name]

        run_script_cmd = ' '.join(cmd_parts)
    else:
        run_script_cmd = './' + script_name

    # print run_script_cmd
    # actual command to run remotely
    remote_cmd = " && ".join([
        "echo \'%s\' > %s" % (script_cmd, script_name),
        "chmod +x %s" % script_name,
        run_script_cmd,
        "rm %s" % script_name])

    return run_on_server(remote_cmd, prompt_for_password = prompt_for_password, **tb_ut.retrieve_values(
        locals(), ['servername', 'username', 'password',
            'folderpath', 'wait_for_output']))

def query_job_states_bridges(username,servername = 'bridges.psc.xsede.org', password = None,jobid = None,jobname=None):
    # If jobid or jobname is specified, look for the specific job and return its status
    # Otherwise, return a dictionary with Key being status, Value being a list of jobs under such state

    lookup = {"PD" : "Pending", "R":"Running", "CA" : "Cancelled", "CF" : "Configuring", 
            "CG":"Completing","CD":"Completed","F":"Failed","TO": "Timeout", "NF": "Node Failure", 
            "RV": "Revoked", "SE" : "Special Exit State", "BF": "Boot Fail", "DL":"Deadline","OOM":"Out Of Memory",
            "S":"Suspended","SI": "Signaling", "ST": "Stopped", "RS": "Resizing", "RQ": "Requeue", "RH": "Requeue Hold",
            "RF":"Requeue Fed: Job is being requeued by a federation","RD":"Resv Del Hold: Job is held"}

    prompt_for_password = (password == None)
    query_cmd = "squeue -u %s" % username
    stdout = run_on_server(query_cmd,servername,username,password=password,prompt_for_password=prompt_for_password)[0]
    stdout = map(lambda x: str(x).strip(),stdout)
    status = {}

    if len(stdout) == 1:
        return status if (jobid == None and jobname == None) else "Not found the specified job."
    
    for i in range(1,len(stdout),2):
        r = stdout[i].split()
        jid = int(r[0])
        partition = r[1]
        ST = r[4]
        jname = r[2]
        if jobid != None and jid == jobid:
            return "Job %d @ Partition: %s @ Status: %s" % (jid,partition,lookup[ST])
        if jobname != None and jname == jobname:
            return "Job '%s' (Job-id: %d) @ Partition: %s @ Status: %s" % (jname,jid,partition,lookup[ST])
        elif jobid == None:
            if lookup[ST] not in status:
                status[lookup[ST]] = []
            status[lookup[ST]].append("Job %s @ Partition: %s" % (jid,partition))
        
    return status if (jobid == None and jobname == None) else "Not found the specified job."

def sync_local_to_remote(src_folderpath, dst_folderpath,
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

def sync_remote_to_local(src_folderpath, dst_folderpath,
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


# print run_on_bridges(**kk)[0]
print query_job_states_bridges("zejiea")

# print sync_local_to_remote("../research_toolbox","~/misc/",'bridges.psc.xsede.org',username='zejiea')
