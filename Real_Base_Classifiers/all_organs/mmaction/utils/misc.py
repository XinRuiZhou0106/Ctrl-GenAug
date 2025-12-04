# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
import random
import string
import inspect
from types import FunctionType, ModuleType
from typing import Optional, Union


def get_random_string(length=15):
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Default: 15.
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id():
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir():
    """Get shm dir for temporary usage."""
    return '/dev/shm'

def get_str_type(module: Union[str, ModuleType, FunctionType]) -> str:
    """Return the string type name of module.

    Args:
        module (str | ModuleType | FunctionType):
            The target module class

    Returns:
        Class name of the module
    """
    if isinstance(module, str):
        str_type = module
    elif inspect.isclass(module) or inspect.isfunction(module):
        str_type = module.__name__
    else:
        return None

    return str_type
