
ut_show_cpu_info() { lscpu; }
ut_show_gpu_info() { nvidia-smi; }
ut_show_memory_info() { free -m; }
ut_show_hardware_info() { lshw; }

ut_get_containing_folderpath() { echo "$(dirname "$1")"; }
ut_get_filename_from_filepath() { echo "$(basename "$1")"; }
ut_get_foldername_from_folderpath() { echo "$(basename "$1")"; }

ut_find_files() { find "$1" -name "$2"; }
ut_find_folders() { find "$1" -type d -name "$2"; }
ut_create_folder() { mkdir -p "$1"; }
ut_copy_folder() { ut_create_folder "$2" && cp -r "$1"/* "$2"; }
ut_rename() { mv "$1" "$2"; }
ut_delete_folder() { rm -rf "$1"; }
ut_delete_folder_interactively() { rm -rfi "$1"; }
ut_get_folder_size() { du -sh $1; }

ut_sleep_in_seconds() { sleep "$1s"; }

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
ut_discard_current_changes_on_git_tracked_file() { git checkout -- "$1"; }

ut_grep_history() { history | grep "$1"; }
ut_show_known_hosts() { cat ~/.ssh/config; }
ut_register_ssh_key_on_server() { ssh-copy-id "$1"; }

ut_create_folder_on_server() { ut_run_command_on_server "mkdir -p \"$1\"" "$2"; }
ut_find_files_and_exec() { find "$1" -name "$2" -exec "$3" {} \; }

UT_RSYNC_FLAGS="--archive --update --recursive --verbose"
ut_sync_folder_to_server() { rsync $UT_RSYNC_FLAGS "$1/" "$2/"; }
ut_sync_folder_from_server() { rsync $UT_RSYNC_FLAGS "$1/" "$2/"; }