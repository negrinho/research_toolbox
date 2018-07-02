

ut_show_cpu_info() { lscpu; }
ut_show_gpu_info() { nvidia-smi; }
ut_show_memory_info() { free -m; }
ut_show_hardware_info() { lshw; }

ut_parent_folder() { echo "$(dirname "$1")"; }

ut_find_files() { find "$1" -name "$2"; }
ut_find_folders() { find "$1" -type d -name "$2"; }
ut_create_folder() { mkdir -p "$1"; }
ut_copy_folder_inside() { ut_create_folder "$2" && cp -r "$1" "$2"; }
ut_move_folder_inside() { ut_create_folder "$2" && mv "$1" "$2"; }
ut_rename() { mv "$1" "$2"; }
ut_delete_folder() { rm -rf "$1"; }
ut_delete_folder_interactively() { rm -rfi "$1"; }

ut_sleep_in_seconds() { sleep "$1s"; }

ut_run_on_bridges() { ssh "$1" -t "$2"; }


# run_on_server
# sync_to_server
# sync_from_server
# run_locally
# rsync.

# TODO: other relevant command for running experimental workflows.
# TODO: some small python scripts.