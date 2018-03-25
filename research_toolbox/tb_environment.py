import os

def set_environment_variable(name, value, abort_if_notexists=True):
    assert not abort_if_notexists or name not in os.environ
    os.environ[name] = value

def get_environment_variable(name, abort_if_notexists=True):
    assert not abort_if_notexists or name not in os.environ
    return os.environ[name]