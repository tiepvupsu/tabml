from distutils.core import setup

import setuptools

setup(
    name="tabml",
    packages=setuptools.find_packages(),
    version="0.1.13",
    license="apache-2.0",
    description="A package for machine learning with tabular data",
    author="Tiep Vu",
    author_email="vuhuutiep@gmail.com",
    url="https://github.com/tiepvupsu/tabml",
    download_url="https://github.com/tiepvupsu/tabml/archive/refs/tags/v0_1_13.tar.gz",
    keywords=["Machine Learning", "Tabular"],
    install_requires=[
        "catboost>=0.26.1",
        "GPUtil>=1.4.0",
        "kaggle>=1.5.6",
        "lightgbm>=2.3.1",
        "loguru>=0.5.1",
        "mlflow>=1.20.1",
        "numpy>=1.18.5",
        "pandas>=1.3.1",
        "pandas-profiling>=2.9.0",
        "pydantic>=1.8.2",
        "pyyaml>=5.4.1",
        "scikit-learn>=0.24.2",
        "scipy>=1.6.2",
        "shap>=0.39.0",
        "termgraph>=0.4.2",
        "xgboost>=1.4.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
    ],
)
