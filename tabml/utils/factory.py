import importlib


def create(cls):
    """Creates a class based on its name."""
    module_name, class_name = cls.rsplit(".", 1)
    somemodule = importlib.import_module(module_name)
    cls_instance = getattr(somemodule, class_name)

    return cls_instance
