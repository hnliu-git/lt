from functools import wraps
import os


def load_resource_file_once(cls):
    """Decorator that you can use for a class that is instantiated from a
    resource file to make sure each file is only loaded once. If the file is
    already loaded, the previously instantiated object will be returned.

    The wrapped class needs to have an __init__ function that takes only the
    resource file path as argument."""

    _loaded_resources = {}

    @wraps(cls)
    def get_resource(resource_file):
        abspath = os.path.abspath(resource_file)
        if abspath not in _loaded_resources:
            _loaded_resources[abspath] = cls(abspath)
        return _loaded_resources[abspath]

    return get_resource
