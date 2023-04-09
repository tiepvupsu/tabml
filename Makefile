SHELL = /bin/bash

# Environment
venv:
	python3 -m venv tabml_env
	source tabml_env/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .


# Style
style:
	black .
	flake8
	python3 -m isort -rc tabml/

# Test
test:
	flake8 ./tabml
	mypy --show-traceback ./tabml # see configs in /mypy.ini
	python -m pytest -s --durations=0 --disable-warnings ./
