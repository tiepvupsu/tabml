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
    install_requires=[
        "GitPython==3.1.13",
        "kaggle==1.5.6",
        "lightgbm==2.3.1",
        "loguru==0.5.1",
        "numpy==1.18.5",
        "pandas==1.1.4",
        "pandas-profiling==2.9.0",
        "protobuf==3.13.0",
        "scikit-learn==0.24.1",
        "scipy==1.6.2",
    ],
)
