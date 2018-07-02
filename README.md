## Motivation for a research toolbox

This repo contains a number of Python tools that I developed for doing experimental research in machine learning.
It includes a broad set of functionality:
* file system manipulation, e.g., creating and deleting a file or a folder, checking if a given file or folder exists, and listing files or folders in a directory.
* interacting with a remote server, e.g., synchronizing files and folders to and from the server, and running commands on the server from the local machine.
* writing and reading simple file types, such as JSON, and pickle files.
* creating folders with experiment configurations that can then be easily ran locally or on the server.
* logging functionality for keeping track of important information when running code, e.g., memory usage or time since start.

While Python has a broad set of functionality, using directly this functionality has an unnecessary high cognitive load because the functions necessary to implement the desired functionality are spread across multiple libraries and use different API and different design principles.
Existing APIs are often unnecessarily flexible for the most common use-cases needed by a particular user.
Developing your own wrapper APIs reduces cognitive load by making common use-cases more explicit.
These APIs are easy to use because they are high-level, coherent, and adjusted to the needs of that particular user.
These wrapper APIs can include high-level error-checking for each use-case, which would require considerably higher cognitive load to implement from scratch using existing APIs.

I am not claiming that this library solves all problems that you may have.
I am suggesting that creating and maintaining your own research toolbox is convenient and should lead to being able to get things done faster and an overall more pleasant experience.
I recommend extending this toolbox or develop your own to suit your needs.
This library is work in progress.
The ultimate goal is to go from research idea to results as fast as possible.

## File description
* [tb_augmentation.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_augmentation.py):
    simple data augmentation.
* [tb_data.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_data.py):
    data loaders and data related functionality.
* [tb_debugging.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_debugging.py):
    error checking and debugging functionality .
* [tb_experiments.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_experiments.py):
    writing experiment folders with configurations for running different experiments.
* [tb_filesystem.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_filesystem.py):
    creating, copying, and testing for existence of files and folders.
* [tb_interact.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_interact.py):
    interactive commands for running jobs on the server or locally.
* [tb_io.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_io.py):
    reading and writing simple file types.
* [tb_logging.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_logging.py):
    common logging funtionality.
* [tb_plotting.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_plotting.py):
    wrappers around plotting libraries such as matplotlib to make simple plot generation easier.
* [tb_preprocessing.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_preprocessing.py):
    simple preprocessing functionality for going from raw data to data that is more ameanable for the application of machine learning.
* [tb_project.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_project.py):
    creation of the typical project structure for a machine learning project.
* [tb_random.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_random.py):
    simple random functionality for shuffling, sorting, and sampling.
* [tb_remote.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_remote.py):
    interaction with remote servers, such as syncing folders to and from the local machine, and submitting jobs to the server.
* [tb_resource.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_resource.py):
    getting information about available resources in a machine, such as the number of CPUs or GPUS.
* [tb_training.py](https://github.com/negrinho/research_toolbox/blob/master/research_toolbox/tb_training.py):
    learning rate schedules and additional logic that is often employed in training machine learning models such as saving and loading the best model found during training.

## Example code
```python
### retrieving certain keys from a dictionary (example from tb_utils.py)
def subset_dict_via_selection(d, ks):
    return {k : d[k] for k in ks}

### sorting and randomness tools (examples from tb_random.py)
def argsort(xs, fns, increasing=True):
    """The functions in fns are used to compute a key which are then used to
    construct a tuple which is then used to sort. The earlier keys are more
    important than the later ones.
    """
    def key_fn(x):
        return tuple([f(x) for f in fns])

    idxs, _ = tb_ut.zip_toggle(
        sorted(enumerate(xs),
            key=lambda x: key_fn(x[1]),
            reverse=not increasing))
    return idxs

def sort(xs, fns, increasing=True):
    idxs = argsort(xs, fns, increasing)
    return apply_permutation(xs, idxs)

def apply_permutation(xs, idxs):
    assert len(set(idxs).intersection(range(len(xs)))) == len(xs)
    return [xs[i] for i in idxs]

def apply_inverse_permutation(xs, idxs):
    assert len(set(idxs).intersection(range(len(xs)))) == len(xs)

    out_xs = [None] * len(xs)
    for i_from, i_to in enumerate(idxs):
        out_xs[i_to] = xs[i_from]
    return out_xs

def shuffle_tied(xs_lst):
    assert len(xs_lst) > 0 and len(map(len, xs_lst)) == 1

    n = len(xs_lst[0])
    idxs = random_permutation(n)
    ys_lst = [apply_permutation(xs, idxs) for xs in xs_lst]
    return ys_lst

### io tools (examples from tb_io.py)
def read_textfile(filepath, strip=True):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if strip:
            lines = [line.strip() for line in lines]
        return lines

def write_textfile(filepath, lines, append=False, with_newline=True):
    mode = 'a' if append else 'w'

    with open(filepath, mode) as f:
        for line in lines:
            f.write(line)
            if with_newline:
                f.write("\n")

def read_jsonfile(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
        return d

def write_jsonfile(d, filepath, sort_keys=False):
    with open(filepath, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=sort_keys)

def read_picklefile(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def write_picklefile(x, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(x, f)

### path tools (examples from tb_filesystem.py)
def path_prefix(path):
    return os.path.split(path)[0]

def path_last_element(path):
    return os.path.split(path)[1]

def path_relative_to_absolute(path):
    return os.path.abspath(path)

def path_exists(path):
    return os.path.exists(path)

def file_exists(path):
    return os.path.isfile(path)

def folder_exists(path):
    return os.path.isdir(path)

def create_file(filepath,
        abort_if_exists=True, create_parent_folders=False):
    assert create_parent_folders or folder_exists(path_prefix(filepath))
    assert not (abort_if_exists and file_exists(filepath))

    if create_parent_folders:
        create_folder(path_prefix(filepath),
            abort_if_exists=False, create_parent_folders=True)

    with open(filepath, 'w'):
        pass

def create_folder(folderpath,
        abort_if_exists=True, create_parent_folders=False):
    assert not file_exists(folderpath)
    assert create_parent_folders or folder_exists(path_prefix(folderpath))
    assert not (abort_if_exists and folder_exists(folderpath))

    if not folder_exists(folderpath):
        os.makedirs(folderpath)

def copy_file(src_filepath, dst_filepath,
        abort_if_dst_exists=True, create_parent_folders=False):
    assert file_exists(src_filepath)
    assert src_filepath != dst_filepath
    assert not (abort_if_dst_exists and file_exists(dst_filepath))

    src_filename = path_last_element(src_filepath)
    dst_folderpath = path_prefix(dst_filepath)
    dst_filename = path_last_element(dst_filepath)

    assert create_parent_folders or folder_exists(dst_folderpath)
    if not folder_exists(dst_folderpath):
        create_folder(dst_folderpath, create_parent_folders=True)

    shutil.copyfile(src_filepath, dst_filepath)

def copy_folder(src_folderpath, dst_folderpath,
        ignore_hidden_files=False, ignore_hidden_folders=False, ignore_file_exts=None,
        abort_if_dst_exists=True, create_parent_folders=False):
    assert folder_exists(src_folderpath)
    assert src_folderpath != dst_folderpath
    assert not (abort_if_dst_exists and folder_exists(dst_folderpath))

    if (not abort_if_dst_exists) and folder_exists(dst_folderpath):
        delete_folder(dst_folderpath, abort_if_nonempty=False)

    pref_dst_fo = path_prefix(dst_folderpath)
    assert create_parent_folders or folder_exists(pref_dst_fo)
    create_folder(dst_folderpath, create_parent_folders=create_parent_folders)

    # create all folders in the destination.
    args = subset_dict_via_selection(locals(),
        ['ignore_hidden_folders', 'ignore_hidden_files'])
    fos = list_folders(src_folderpath, use_relative_paths=True, recursive=True, **args)

    for fo in fos:
        fo_path = join_paths([dst_folderpath, fo])
        create_folder(fo_path, create_parent_folders=True)

    # copy all files to the destination.
    args = subset_dict_via_selection(locals(),
        ['ignore_hidden_folders', 'ignore_hidden_files', 'ignore_file_exts'])
    fis = list_files(src_folderpath, use_relative_paths=True, recursive=True, **args)

    for fi in fis:
        src_fip = join_paths([src_folderpath, fi])
        dst_fip = join_paths([dst_folderpath, fi])
        copy_file(src_fip, dst_fip)

def delete_file(filepath, abort_if_notexists=True):
    assert file_exists(filepath) or (not abort_if_notexists)

    if file_exists(filepath):
        os.remove(filepath)

def delete_folder(folderpath, abort_if_nonempty=True, abort_if_notexists=True):
    assert folder_exists(folderpath) or (not abort_if_notexists)

    if folder_exists(folderpath):
        assert len(os.listdir(folderpath)) == 0 or (not abort_if_nonempty)
        shutil.rmtree(folderpath)
    else:
        assert not abort_if_notexists
```