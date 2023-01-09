SHELL = /bin/bash

# Environment
venv:
    python3 -m venv tabml_env
    source tabml_env/bin/activate && \
    python3 -m pip install pip setuptools wheel && \
    python3 -m pip install -e .