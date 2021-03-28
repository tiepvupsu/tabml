from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="tabml",
    version="0.1",
    description="Machine Learning with Tabular data",
    long_description=readme,
    author="Tiep Vu",
    author_email="vuhuutiep@gmail.com",
    url="https://github.com/tiepvupsu/tabml",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    package_dir = {'': 'tabml'}
)
