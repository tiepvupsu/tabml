from distutils.core import setup
from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="tabml",
    packages=setuptools.find_packages(),
    version="0.2.9",
    license="apache-2.0",
    description="A package for machine learning with tabular data",
    author="Tiep Vu",
    author_email="vuhuutiep@gmail.com",
    url="https://github.com/tiepvupsu/tabml",
    download_url="https://github.com/tiepvupsu/tabml/archive/refs/tags/v0.2.9.tar.gz",
    keywords=["Machine Learning", "Tabular"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "catboost>=1.1",
        "GPUtil>=1.4.0",
        "lightgbm>=2.3.1",
        "loguru>=0.5.1",
        "numpy>=1.21,<1.23.0",
        "pandas>=1.4.3",
        "pydantic>=2",
        "pyyaml>=6.0",
        "scikit-learn>=1.1.2",
        "scipy>=1.9.3",
        "shap>=0.39.0",
        "termgraph>=0.4.2",
        "xgboost>=1.6",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
    ],
)
