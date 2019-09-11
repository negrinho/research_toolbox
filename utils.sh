
ut_show_cpu_info() { lscpu; }
ut_show_gpu_info() { nvidia-smi; }
ut_show_memory_info() { free -m; }
ut_show_hardware_info() { lshw; }

ut_up1() { cd ..; }
ut_up2() { cd ../..; }
ut_up3() { cd ../../..; }
ut_get_last_process_id() { echo "$!"; }
ut_get_last_process_exit_code() { echo "$?"; }
ut_wait_on_process_id() { wait "$1"; }
ut_random_uuid() { python -c "import uuid; print(uuid.uuid4())"; }

ut_get_containing_folderpath() { echo "$(dirname "$1")"; }
ut_get_filename_from_filepath() { echo "$(basename "$1")"; }
ut_get_foldername_from_folderpath() { echo "$(basename "$1")"; }
ut_get_absolute_path_from_relative_path() { realpath "$1"; }

ut_find_files() { find "$1" -name "$2"; }
ut_find_folders() { find "$1" -type d -name "$2"; }
ut_create_folder() { mkdir -p "$1"; }
ut_copy_folder() { ut_create_folder "$2" && cp -r "$1"/* "$2"; }
ut_rename() { mv "$1" "$2"; }
ut_delete_folder() { rm -rf "$1"; }
ut_delete_folder_interactively() { rm -rfi "$1"; }
ut_get_folder_size() { du -sh $1; }
ut_rename_file_in_place(){ folderpath="$(dirname "$1")" && mv "$1" "$(dirname "$1")/$2"; }
ut_rename_folder_in_place(){ folderpath="$(dirname "$1")" && mv "$1" "$(dirname "$1")/$2"; }
ut_compress_folder(){ foldername=`ut_get_foldername_from_folderpath $1` && tar -zcf "$foldername.tar.gz" "$1"; }
ut_uncompress_folder(){ tar -zxf "$1"; }

ut_send_mail_message_with_subject_to_address() { echo "$1" | mail "--subject=$2" "$3"; }
ut_send_mail_message_with_subject_and_attachment_to_address() { echo "$1" | mail "--subject=$2" "--attach=$3" "$4"; }

ut_sleep_in_seconds() { sleep "$1s"; }
ut_run_every_num_seconds() { watch -n "$2" "$1"; }

ut_run_headless_command() { nohup $1; }
ut_run_command_on_server() { ssh "$2" -t "$1"; }
ut_run_command_on_server_on_folder() { ssh "$2" -t "cd \"$3\" && $1"; }
ut_run_bash_on_server_on_folder() { ssh "$1" -t "cd \"$2\" && bash"; }
ut_run_python_command() { python -c "$1" >&2; }
ut_profile_python_with_cprofile() { python -m cProfile -s cumtime $"$1"; }

ut_map_jsonfile() { ut_run_python_command \
    "from research_toolbox.tb_io import read_jsonfile, write_jsonfile;"\
    "fn = $1; write_jsonfile(fn(read_jsonfile(\"$2\")), \"$3\");";
}

ut_get_git_head_sha() { git rev-parse HEAD; }
ut_show_git_commits_for_file(){ git log --follow -- "$1"; }
ut_show_oneline_git_log(){ git log --pretty=oneline; }
ut_show_files_ever_tracked_by_git() { git log --pretty=format: --name-only --diff-filter=A | sort - | sed '/^$/d'; }
ut_show_files_currently_tracked_by_git_on_branch() { git ls-tree -r "$1" --name-only; }
ut_discard_git_uncommited_changes_for_file() { git checkout -- "$1"; }
ut_discard_all_git_uncommitted_changes() { git checkout -- .; }
ut_delete_files_from_git_history(){
    git filter-branch --prune-empty -d ./scratch_git_filter \
  --index-filter "git rm -r --cached -f --ignore-unmatch $1" \
  --tag-name-filter cat -- --all;
  git reflog expire --expire=now --all;
  git gc --prune=now;
}



ut_grep_history() { history | grep "$1"; }
ut_show_known_hosts() { cat ~/.ssh/config; }
ut_register_ssh_key_on_server() { ssh-copy-id "$1"; }

ut_create_folder_on_server() { ut_run_command_on_server "mkdir -p \"$1\"" "$2"; }
ut_find_files_and_exec() { find "$1" -name "$2" -exec "$3" {} \ ; }

UT_RSYNC_FLAGS="--archive --update --recursive --verbose"
ut_sync_folder_to_server() { rsync $UT_RSYNC_FLAGS "$1/" "$2/"; }
ut_sync_folder_from_server() { rsync $UT_RSYNC_FLAGS "$1/" "$2/"; }

ut_show_environment_variables() { printenv; }
ut_preappend_to_pythonpath() { export PYTHONPATH="$1:$PYTHONPATH"; }

ut_run_command_on_server() { ssh "$2" -t "$1"; }
ut_run_command_on_server_on_folder() { ssh "$2" -t "cd \"$3\" && $1"; }
ut_run_bash_on_server_on_folder() { ssh "$1" -t "cd \"$2\" && bash"; }

# both are slurm managed clusters. hosts defined in ~/.ssh/config
ut_run_command_on_bridges() { ut_run_command_on_server "$1" bridges; }
ut_run_command_on_bridges_on_folder() { ut_run_command_on_server_on_folder "$1" bridges "$2"; }
ut_run_bash_on_bridges_on_folder() { ut_run_bash_on_server_on_folder bridges "$1"; }

ut_run_command_on_matrix() { ut_run_command_on_server "$1" matrix; }
ut_run_command_on_matrix_on_folder() { ut_run_command_on_server_on_folder "$1" matrix "$2"; }
ut_run_bash_on_matrix_on_folder() { ut_run_bash_on_server_on_folder matrix "$1"; }

# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
ut_create_conda_environment() { conda create --name "$1"; }
ut_create_conda_py27_environment() { conda create --name "$1" python=2.7; }
ut_create_conda_py36_environment() { conda create --name "$1" python=3.6; }
ut_show_conda_environments() { conda info --envs; }
ut_show_installed_conda_packages() { conda list; }
ut_delete_conda_environment() { conda env remove --name "$1"; }
ut_activate_conda_environment() { conda activate "$1"; }
ut_deactivate_conda_environment() { conda deactivate; }


# command, job name, folder, num cpus, memory in mbs, time in minutes
# limits: 4GB per cpu, 48 hours,
# NOTE: read https://www.psc.edu/bridges/user-guide/running-jobs for more details.
ut_submit_bridges_cpu_job_with_resources() {
    script='#!/bin/bash'"
#SBATCH --nodes=1
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=$4
#SBATCH --mem=$5MB
#SBATCH --time=$6
#SBATCH --job-name=\"$2\"
$1" && ut_run_command_on_bridges "cd \"./$3\" && echo \"$script\" > _run.sh && chmod +x _run.sh && sbatch _run.sh && rm _run.sh";
}

# 1: command, 2: job name, 3: folder, 4: num cpus, 5: num_gpus, 6: memory in mbs, 7: time in minutes
# limits: 7GB per gpu, 48 hours, 16 cores per gpu,
# NOTE: read https://www.psc.edu/bridges/user-guide/running-jobs for more details.
ut_submit_bridges_gpu_job_with_resources() {
    script='#!/bin/bash'"
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:k80:$5
#SBATCH --cpus-per-task=$4
#SBATCH --mem=$6MB
#SBATCH --time=$7
#SBATCH --job-name=\"$2\"
$1" && ut_run_command_on_bridges "cd \"./$3\" && echo \"$script\" > _run.sh && chmod +x _run.sh && sbatch _run.sh && rm _run.sh";
}

ut_show_bridges_queue() { ut_run_command_on_bridges "squeue"; }
ut_show_my_jobs_on_bridges() { ut_run_command_on_bridges "squeue -u rpereira"; }
ut_cancel_job_on_bridges() { ut_run_command_on_bridges "scancel -n \"$1\""; }
ut_cancel_all_my_jobs_on_bridges() { ut_run_command_on_bridges "scancel -u rpereira"; }

# uses a deepo docker based image.
ut_build_py27_cpu_singularity_container() { sudo singularity build --sandbox py27_cpu.img docker://ufoym/deepo:all-py27-cpu; }
ut_build_py36_cpu_singularity_container() { sudo singularity build --sandbox py36_cpu.img docker://ufoym/deepo:all-py36-cpu; }
ut_build_py27_gpu_singularity_container() { sudo singularity build --sandbox py27_gpu.img docker://ufoym/deepo:all-py27; }
ut_build_py36_gpu_singularity_container() { sudo singularity build --sandbox py36_gpu.img docker://ufoym/deepo:all-py36; }

ut_build_writable_singularity_container_from_recipe(){ sudo singularity build --sandbox "$1" "$2"; }
ut_build_static_singularity_container_from_recipe(){ sudo singularity build "$1" "$2"; }
ut_build_static_singularity_container_from_writable_singularity_container(){ sudo singularity build "$1" "$2"; }
# NOTE: using the --nv flag to check for nvidia binaries by default for simplicity. ignore warning if they don't exist.
ut_bash_into_singularity_container(){ singularity shell --nv "$1"; }
ut_sudo_bash_into_singularity_container(){ sudo singularity shell --nv --writable "$1"; }
ut_run_command_in_singularity_container(){ singularity exec  --nv "$1" "$2"; }

# NOTE: the bind command allows to expose directories in the host operating
# system inside the container.
# multiple paths can be bound by separating them with commas.
# singularity shell --bind /path/to/dir/host0:/path/to/dir/container0,/path/to/dir/host1:/path/to/dir/container1 my_container.sif
# See docs: https://www.sylabs.io/guides/3.0/user-guide/index.html

ut_install_packages() {
    sudo apt-get install \
        singularity \
        mailutils \
        tree;
}
