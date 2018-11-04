### directory management.
import os
import shutil
import uuid
import research_toolbox.tb_utils as tb_ut


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


def create_file(filepath, abort_if_exists=True, create_parent_folders=False):
    assert create_parent_folders or folder_exists(path_prefix(filepath))
    assert not (abort_if_exists and file_exists(filepath))

    if create_parent_folders:
        create_folder(
            path_prefix(filepath),
            abort_if_exists=False,
            create_parent_folders=True)

    with open(filepath, 'w'):
        pass


def create_folder(folderpath, abort_if_exists=True,
                  create_parent_folders=False):
    assert not file_exists(folderpath)
    assert create_parent_folders or folder_exists(path_prefix(folderpath))
    assert not (abort_if_exists and folder_exists(folderpath))

    if not folder_exists(folderpath):
        os.makedirs(folderpath)


def copy_file(src_filepath,
              dst_filepath,
              abort_if_dst_exists=True,
              create_parent_folders=False):
    assert file_exists(src_filepath)
    assert src_filepath != dst_filepath
    assert not (abort_if_dst_exists and file_exists(dst_filepath))

    dst_folderpath = path_prefix(dst_filepath)
    assert create_parent_folders or folder_exists(dst_folderpath)
    if not folder_exists(dst_folderpath):
        create_folder(dst_folderpath, create_parent_folders=True)

    shutil.copyfile(src_filepath, dst_filepath)


def copy_folder(src_folderpath,
                dst_folderpath,
                ignore_hidden_files=False,
                ignore_hidden_folders=False,
                ignore_file_exts=None,
                abort_if_dst_exists=True,
                create_parent_folders=False):
    assert folder_exists(src_folderpath)
    assert src_folderpath != dst_folderpath
    assert not (abort_if_dst_exists and folder_exists(dst_folderpath))

    if (not abort_if_dst_exists) and folder_exists(dst_folderpath):
        delete_folder(dst_folderpath, abort_if_nonempty=False)

    pref_dst_fo = path_prefix(dst_folderpath)
    assert create_parent_folders or folder_exists(pref_dst_fo)
    create_folder(dst_folderpath, create_parent_folders=create_parent_folders)

    # create all folders in the destination.
    fos = list_folders(
        src_folderpath,
        use_relative_paths=True,
        recursive=True,
        ignore_hidden_folders=ignore_hidden_folders)

    for fo in fos:
        fo_path = join_paths([dst_folderpath, fo])
        create_folder(fo_path, create_parent_folders=True)

    # copy all files to the destination.
    kwargs = tb_ut.subset_dict_via_selection(
        locals(),
        ['ignore_hidden_folders', 'ignore_hidden_files', 'ignore_file_exts'])
    fis = list_files(
        src_folderpath, use_relative_paths=True, recursive=True, **kwargs)
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


def list_paths(folderpath,
               ignore_files=False,
               ignore_dirs=False,
               ignore_hidden_folders=True,
               ignore_hidden_files=True,
               ignore_file_exts=None,
               recursive=False,
               use_relative_paths=False):
    assert folder_exists(folderpath)

    path_list = []
    # enumerating all desired paths in a directory.
    for root, dirs, files in os.walk(folderpath):
        if ignore_hidden_folders:
            dirs[:] = [d for d in dirs if not d[0] == '.']
        if ignore_hidden_files:
            files = [f for f in files if not f[0] == '.']
        if ignore_file_exts != None:
            files = [
                f for f in files
                if not any([f.endswith(ext) for ext in ignore_file_exts])
            ]

        # get only the path relative to this path.
        if not use_relative_paths:
            pref_root = root
        else:
            pref_root = os.path.relpath(root, folderpath)

        if not ignore_files:
            path_list.extend([join_paths([pref_root, f]) for f in files])
        if not ignore_dirs:
            path_list.extend([join_paths([pref_root, d]) for d in dirs])

        if not recursive:
            break
    return path_list


def list_files(folderpath,
               ignore_hidden_folders=True,
               ignore_hidden_files=True,
               ignore_file_exts=None,
               recursive=False,
               use_relative_paths=False):

    kwargs = tb_ut.subset_dict_via_selection(locals(), [
        'recursive', 'ignore_hidden_folders', 'ignore_hidden_files',
        'ignore_file_exts', 'use_relative_paths'
    ])
    return list_paths(folderpath, ignore_dirs=True, **kwargs)


def list_folders(folderpath,
                 ignore_hidden_folders=True,
                 recursive=False,
                 use_relative_paths=False):

    kwargs = tb_ut.subset_dict_via_selection(
        locals(), ['recursive', 'ignore_hidden_folders', 'use_relative_paths'])
    return list_paths(folderpath, ignore_files=True, **kwargs)


def list_leaf_folders(root_folderpath, ignore_hidden_folders=True):

    def iter_fn(folderpath, leaf_folderpath_lst):
        child_folderpath_lst = list_folders(
            folderpath, ignore_hidden_folders=ignore_hidden_folders)

        if len(child_folderpath_lst) > 0:
            for p in child_folderpath_lst:
                iter_fn(p, leaf_folderpath_lst)
        else:
            leaf_folderpath_lst.append(folderpath)
        return leaf_folderpath_lst

    return iter_fn(root_folderpath, [])


# NOTE: I think that this function is more general than list_leaf_folders.
def list_folders_conditionally(root_folderpath,
                               cond_fn,
                               ignore_hidden_folders=True):
    """Descends down the directory tree rooted at the specified folder path, tests a
    condition at each node, and stops the descent down that particular path if
    the condition evaluates to true. The nodes which evaluated to true are
    returned in a list.

    This function is useful, for example, to work with arbitrarily nested folders
    that always end up having some regular folder structure close to the leaves.
    """

    def iter_fn(folderpath, lst):

        if cond_fn(folderpath):
            lst.append(folderpath)
        else:
            child_folderpath_lst = list_folders(
                folderpath, ignore_hidden_folders=ignore_hidden_folders)
            for p in child_folderpath_lst:
                iter_fn(p, lst)
        return lst

    return iter_fn(root_folderpath, [])


def join_paths(paths):
    return os.path.join(*paths)


def pairs_to_filename(ks, vs, kv_sep='', pair_sep='_', prefix='',
                      suffix='.txt'):
    pairs = [kv_sep.join([k, v]) for (k, v) in zip(ks, vs)]
    s = prefix + pair_sep.join(pairs) + suffix
    return s


def get_unique_filename(folderpath, fileext):
    while True:
        filename = uuid.uuid4()
        if not file_exists(
                join_paths([folderpath,
                            "%s.%s" % (filename, fileext)])):
            return filename


def get_unique_filepath(folderpath, fileext):
    filename = get_unique_filename(folderpath, fileext)
    return join_paths([folderpath, "%s.%s" % (filename, fileext)])


def get_current_working_directory():
    return os.getcwd()