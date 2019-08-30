from logging import warning
from tempfile import mkstemp, NamedTemporaryFile


def create_tempfile_preferably_in_dir(suffix=None, prefix=None, dir=None, text=False, persistant: bool = False, ):
    """ Create a temporary file with precedence for dir if possible, in TMP otherwise.
    For example, this is useful to try to save into /dev/shm.
    """
    temp_file = mkstemp if persistant else NamedTemporaryFile
    try:
        handle = temp_file(suffix=suffix, prefix=prefix, dir=dir)
    except FileNotFoundError:
        handle = temp_file(suffix=suffix, prefix=prefix, dir=None)
        warning(f'Could not create temp file in {dir}. '
                f'Instead, the path is {handle}.')
    try:
        path = handle.name
    except AttributeError:
        _, path = handle
    return path
