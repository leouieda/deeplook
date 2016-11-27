from __future__ import absolute_import, division, print_function
import os


_backends = {'numpy'}
_default_backend = "numpy"
_env_variable = "DEEPLOOK_BACKEND"


def get_backend():
    if _env_variable in os.environ:
        backend = os.environ[_env_variable]
        assert backend in _backends, "Invalid backend {}.".format(backend)
    else:
        backend = _default_backend
    return backend


def set_backend(backend):
    assert backend in _backends, "Invalid backend {}.".format(backend)
    os.environ[_env_variable] = backend


def backend():
    return _backend


_BACKEND = get_backend()
if _BACKEND == 'numpy':
    from .numpy_backend import *
else:
    raise Exception("Invalid backend {}.".format(_BACKEND)
