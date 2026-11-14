import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class Registry(object):
    """Map configuration names to registered component classes."""

    def __init__(self):
        self.data = {}

    def register_module(self, module_name=None):
        """Create a decorator that stores a component class under its lookup name.

        Args:
            module_name: Explicit lookup key, or None to use the class name.

        Returns:
            Class decorator that registers and returns the original class.
        """
        def _register(cls):
            name = module_name
            if module_name is None:
                name = cls.__name__
            self.data[name] = cls
            return cls

        return _register

    def __getitem__(self, key):
        return self.data[key]


BACKBONE = Registry()
DETECTOR = Registry()
LOSSFUNC = Registry()


# Import components after creating registries so decorators can register during package imports.
import backbones
import detectors
import losses
