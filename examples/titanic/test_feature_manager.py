import os

from . import feature_manager

# import pytest


def test_run(request):
    os.chdir(request.fspath.dirname)
    feature_manager.run()
    os.chdir(request.config.invocation_dir)
