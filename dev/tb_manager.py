

import json
import uuid
import research_toolbox.tb_filesystem as tb_fs
import research_toolbox.tb_io as tb_io
from six import itervalues

# NOTE: talk about doing things lazily.
# does not support multiprocess right now, but it is not a major concern due
# to the way that the files are generated.
# NOTE: this is perhaps a bit too complicated.
# it would be better to have just the get memo, and do something that works.
# only functions that have config_lst should be (some?) of the memo functions.
# the remaining functions can be done with a single config.
class NestedMemoManager:
    def __init__(self, folderpath, create_if_notexists=False):
        self.folderpath = folderpath
        self.key_to_filename = {}
        self.key_to_foldername = {}
        self.key_to_memo = {}

        tb_fs.create_folder(folderpath,
            abort_if_exists=False,
            create_parent_folders=create_if_notexists)

        # initialize the memo based on the state of the memo folder:
        for p in tb_fs.list_files(folderpath):
            name_with_ext = tb_fs.path_last_element(p)
            # for the files.
            if name_with_ext.startswith('file_config-') and name_with_ext.endswith('.json'):
                name = name_with_ext[len('file_config-'):-len('.json')]
                config = tb_io.read_jsonfile(p)
                key = self._key_from_config(config)
                self.key_to_filename[key] = name

            # for the sub-memos.
            elif name_with_ext.startswith('memo_config-') and name_with_ext.endswith('.json'):
                name = name_with_ext[len('memo_config-'):-len('.json')]
                config = tb_io.read_jsonfile(p)
                key = self._key_from_config(config)
                self.key_to_foldername[key] = name

    def _get_filepath(self, filetype, filename, fileext):
        return tb_fs.join_paths([self.folderpath, "%s-%s.%s" % (filetype, filename, fileext)])

    def _get_folderpath(self, foldername):
        return tb_fs.join_paths([self.folderpath, "memo_value-%s" % foldername])

    # auxiliary function used to retrieve sub-memos.
    def _get_memo(self, config_lst, create_if_notexists=False):
        memo = self
        for cfg in config_lst:
            key = memo._key_from_config(cfg)
            if key not in memo.key_to_foldername:
                if create_if_notexists:
                    foldername = memo._get_unique_foldername()
                    value_folderpath = memo._get_folderpath(foldername)
                    cfg_filepath = memo._get_filepath("memo_config", foldername, "json")
                    tb_fs.create_folder(value_folderpath)
                    tb_io.write_jsonfile(cfg, cfg_filepath)

                    next_memo = NestedMemoManager(value_folderpath)
                    memo.key_to_foldername[key] = foldername
                    memo.key_to_memo[key] = next_memo
                else:
                    return None
            else:
                if key not in memo.key_to_memo:
                    foldername = memo.key_to_foldername[key]
                    value_folderpath = memo._get_folderpath(foldername)
                    memo.key_to_memo[key] = NestedMemoManager(value_folderpath)
                next_memo = memo.key_to_memo[key]
            memo = next_memo
        return memo

    def _key_from_config(self, config):
        return json.dumps(config, sort_keys=True)

    def _get_unique_name(self, prefix):
        while True:
            filename = uuid.uuid4()
            filepath = tb_fs.join_paths([self.folderpath, "%s-%s.json" % (prefix, filename)])
            if not tb_fs.file_exists(filepath):
                return filename

    def _get_unique_filename(self):
        return self._get_unique_name("file_config")

    def _get_unique_foldername(self):
        return self._get_unique_name("memo_config")

    def is_file_available(self, config_lst):
        assert len(config_lst) > 0
        memo = self._get_memo(config_lst[:-1], create_if_notexists=False)
        if memo is None:
            return False
        else:
            key = self._key_from_config(config_lst[-1])
            return key in memo.key_to_filename

    def is_memo_available(self, config_lst):
        assert len(config_lst) > 0
        return self._get_memo(config_lst, create_if_notexists=False) is not None

    def get_memo(self, config_lst, create_parent_memos=False):
        assert len(config_lst) > 0
        memo = self._get_memo(config_lst, create_if_notexists=create_parent_memos)
        assert memo is not None
        return memo

    def create_memo(self, config_lst):
        assert not self.is_memo_available(config_lst)
        memo = self._get_memo(config_lst, create_if_notexists=True)
        return memo

    def delete_memo(self, config_lst, abort_if_notexists=True):
        assert len(config_lst) > 0

        parent_memo = self._get_memo(config_lst[:-1])
        if parent_memo.is_memo_available(config_lst[-1:]):
            key = parent_memo._key_from_config(config_lst[-1])
            foldername = parent_memo.key_to_foldername[key]
            cfg_filepath = parent_memo._get_filepath("memo_config", foldername, "json")
            memo_folderpath = parent_memo._get_folderpath(foldername)
            tb_fs.delete_file(cfg_filepath)
            tb_fs.delete_folder(memo_folderpath)
            parent_memo.key_to_foldername.pop(key)
            parent_memo.key_to_memo.pop(key)
        else:
            assert not abort_if_notexists

    def read_file(self, config_lst):
        assert len(config_lst) > 0
        memo = self._get_memo(config_lst[:-1])
        key = memo._key_from_config(config_lst[-1])
        filename = memo.key_to_filename[key]
        value_filepath = memo._get_filepath('file_value', filename, 'pkl')
        return tb_io.read_picklefile(value_filepath)

    def write_file(self, config_lst, value, abort_if_exists=True):
        assert len(config_lst) > 0
        memo = self._get_memo(config_lst[:-1])
        assert memo is not None
        cfg = config_lst[-1]
        key = memo._key_from_config(cfg)
        assert not abort_if_exists or key not in memo.key_to_filename

        # if it exists, get it from the dictionary.
        if key in memo.key_to_filename:
            filename = memo.key_to_filename[key]
        else:
            filename = memo._get_unique_filename()

        config_filepath = memo._get_filepath('file_config', filename, 'json')
        tb_io.write_jsonfile(cfg, config_filepath)
        value_filepath = memo._get_filepath('file_value', filename, 'pkl')
        tb_io.write_picklefile(value, value_filepath)
        memo.key_to_filename[key] = filename

    def delete_file(self, config_lst, abort_if_notexists=True):
        assert len(config_lst) > 0
        memo = self._get_memo(config_lst[:-1])
        assert memo is not None
        cfg = config_lst[-1]
        key = memo._key_from_config(cfg)
        if key in memo.key_to_filename:
            filename = memo.key_to_filename.pop(key)
            cfg_filepath = memo._get_filepath("file_config", filename, "json")
            value_filepath = memo._get_filepath("file_value", filename, "pkl")
            tb_fs.delete_file(cfg_filepath)
            tb_fs.delete_file(value_filepath)
        else:
            assert not abort_if_notexists

    def get_file_configs(self):
        lst = []
        for filename in itervalues(self.key_to_filename):
            filepath = self._get_filepath("file_config", filename, "json")
            cfg = tb_io.read_jsonfile(filepath)
            lst.append(cfg)
        return lst

    def get_memo_configs(self):
        lst = []
        for foldername in itervalues(self.key_to_foldername):
            filepath = self._get_filepath("memo_config", foldername, "json")
            cfg = tb_io.read_jsonfile(filepath)
            lst.append(cfg)
        return lst

# sligthly simplified to reduce the amount of repetition in the reading and
# writing operations.
class SimplifiedNestedMemoManager:
    def __init__(self, folderpath, create_if_notexists=False):
        self.folderpath = folderpath
        self.key_to_filename = {}
        self.key_to_foldername = {}
        self.key_to_memo = {}

        tb_fs.create_folder(folderpath,
            abort_if_exists=False,
            create_parent_folders=create_if_notexists)

        # initialize the memo based on the state of the memo folder:
        for p in tb_fs.list_files(folderpath):
            name_with_ext = tb_fs.path_last_element(p)
            # for the files.
            if name_with_ext.startswith('file_config-') and name_with_ext.endswith('.json'):
                name = name_with_ext[len('file_config-'):-len('.json')]
                config = tb_io.read_jsonfile(p)
                key = self._key_from_config(config)
                self.key_to_filename[key] = name

            # for the sub-memos.
            elif name_with_ext.startswith('memo_config-') and name_with_ext.endswith('.json'):
                name = name_with_ext[len('memo_config-'):-len('.json')]
                config = tb_io.read_jsonfile(p)
                key = self._key_from_config(config)
                self.key_to_foldername[key] = name

    def _get_filepath(self, filetype, filename, fileext):
        return tb_fs.join_paths([self.folderpath, "%s-%s.%s" % (filetype, filename, fileext)])

    def _get_folderpath(self, foldername):
        return tb_fs.join_paths([self.folderpath, "memo_value-%s" % foldername])

    def _get_file_paths(self, filename):
        cfg_filepath = tb_fs.join_paths([self.folderpath, "file_config-%s.json" % filename])
        value_filepath = tb_fs.join_paths([self.folderpath, "file_value-%s.pkl" % filename])
        return (cfg_filepath, value_filepath)

    def _get_memo_paths(self, foldername):
        cfg_filepath = tb_fs.join_paths([self.folderpath, "memo_config-%s.json" % foldername])
        memo_folderpath = tb_fs.join_paths([self.folderpath, "memo_value-%s" % foldername])
        return (cfg_filepath, memo_folderpath)

    # auxiliary function used to retrieve sub-memos.
    def _get_memo(self, config_lst, create_if_notexists=False):
        memo = self
        for cfg in config_lst:
            key = memo._key_from_config(cfg)
            if key not in memo.key_to_foldername:
                if create_if_notexists:
                    # get new unique name
                    foldername = memo._get_unique_foldername()
                    cfg_filepath, memo_folderpath = memo._get_memo_paths(foldername)
                    tb_fs.create_folder(memo_folderpath)
                    tb_io.write_jsonfile(cfg, cfg_filepath)

                    next_memo = SimplifiedNestedMemoManager(memo_folderpath)
                    memo.key_to_foldername[key] = foldername
                    memo.key_to_memo[key] = next_memo
                else:
                    return None
            else:
                if key not in memo.key_to_memo:
                    # use existing name
                    foldername = memo.key_to_foldername[key]
                    memo_folderpath = memo._get_memo_paths(foldername)[1]
                    memo.key_to_memo[key] = SimplifiedNestedMemoManager(memo_folderpath)
                next_memo = memo.key_to_memo[key]
            memo = next_memo
        return memo

    def _key_from_config(self, config):
        return json.dumps(config, sort_keys=True)

    def _get_unique_name(self, prefix):
        while True:
            filename = uuid.uuid4()
            filepath = tb_fs.join_paths([self.folderpath, "%s-%s.json" % (prefix, filename)])
            if not tb_fs.file_exists(filepath):
                return filename

    def _get_unique_filename(self):
        return self._get_unique_name("file_config")

    def _get_unique_foldername(self):
        return self._get_unique_name("memo_config")

    def is_file_available(self, config):
        key = self._key_from_config(config)
        return key in self.key_to_filename

    def is_memo_available(self, config_lst):
        assert len(config_lst) > 0
        return self._get_memo(config_lst, create_if_notexists=False) is not None

    def get_memo(self, config_lst):
        assert len(config_lst) > 0
        memo = self._get_memo(config_lst, create_if_notexists=False)
        assert memo is not None
        return memo

    def create_memo(self, config_lst):
        assert not self.is_memo_available(config_lst)
        memo = self._get_memo(config_lst, create_if_notexists=True)
        return memo

    def delete_memo(self, config_lst, abort_if_notexists=True):
        assert len(config_lst) > 0

        parent_memo = self._get_memo(config_lst[:-1])
        if parent_memo.is_memo_available(config_lst[-1:]):
            key = parent_memo._key_from_config(config_lst[-1])
            foldername = parent_memo.key_to_foldername[key]
            cfg_filepath, memo_folderpath = parent_memo._get_memo_paths(foldername)
            tb_fs.delete_file(cfg_filepath)
            tb_fs.delete_folder(memo_folderpath)
            parent_memo.key_to_foldername.pop(key)
            parent_memo.key_to_memo.pop(key)
        else:
            assert not abort_if_notexists

    def read_file(self, config):
        key = self._key_from_config(config)
        filename = self.key_to_filename[key]
        value_filepath = self._get_file_paths(filename)[1]
        return tb_io.read_picklefile(value_filepath)

    def write_file(self, config, value, abort_if_exists=True):
        key = self._key_from_config(config)
        assert not abort_if_exists or key not in self.key_to_filename

        # if it exists, get it from the dictionary.
        if key not in self.key_to_filename:
            filename = self._get_unique_filename()
        filename = self.key_to_filename[key]
        cfg_filepath, value_filepath = self._get_file_paths(filename)
        tb_io.write_jsonfile(config, cfg_filepath)
        tb_io.write_picklefile(value, value_filepath)

    def delete_file(self, config, abort_if_notexists=True):
        key = self._key_from_config(config)
        if key in self.key_to_filename:
            filename = self.key_to_filename.pop(key)
            cfg_filepath, value_filepath = self._get_file_paths(filename)
            tb_fs.delete_file(cfg_filepath)
            tb_fs.delete_file(value_filepath)
        else:
            assert not abort_if_notexists

    def get_file_configs(self):
        lst = []
        for filename in itervalues(self.key_to_filename):
            filepath = self._get_file_paths(filename)[0]
            cfg = tb_io.read_jsonfile(filepath)
            lst.append(cfg)
        return lst

    def get_memo_configs(self):
        lst = []
        for foldername in itervalues(self.key_to_foldername):
            filepath = self._get_memo_paths(foldername)[0]
            cfg = tb_io.read_jsonfile(filepath)
            lst.append(cfg)
        return lst