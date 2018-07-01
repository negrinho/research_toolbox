

ut_show_cpu_info() { lscpu; }
ut_show_gpu_info() { nvidia-smi; }
ut_show_memory_info() { free -m; }
ut_show_hardware_info() { lshw; }

ut_parent_folder() { echo "$(dirname "$1")"; }

ut_find_files() { find "$1" -name "$2"; }
ut_find_folders() { find "$1" -type d -name "$2"; }

ut_sleep_in_seconds() { sleep $1; }


# run_on_server
# sync_to_server
# sync_from_server
# run_locally

# TODO: other relevant command for running experimental workflows.