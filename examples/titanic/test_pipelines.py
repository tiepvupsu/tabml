import os

from . import pipelines


def test_full_pipeline(request):
    os.chdir(request.fspath.dirname)
    pipelines.train_lgbm()
    os.chdir(request.config.invocation_dir)
