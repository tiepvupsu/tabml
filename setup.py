from distutils.core import setup

import setuptools

setup(
    name="tabml",  # How you named your package folder (MyLib)
    packages=setuptools.find_packages(),
    version="0.1.3",  # Start with a small number and increase it with every change you make
    license="apache-2.0",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="A package for machine learning with tabular data",  # Give a short description about your library
    author="Tiep Vu",  # Type in your name
    author_email="vuhuutiep@gmail.com",  # Type in your E-Mail
    url="https://github.com/tiepvupsu/tabml",  # Provide either the link to your github or to your website
    download_url="https://github.com/tiepvupsu/tabml/archive/refs/tags/v_01.tar.gz",  # I explain this later on
    keywords=["Machine Learning", "Tabular"],  # Keywords that define your package best
    install_requires=[
        "GitPython>=3.1.13",
        "GPUtil>=1.4.0",
        "kaggle>=1.5.6",
        "lightgbm>=2.3.1",
        "loguru>=0.5.1",
        "numpy>=1.18.5",
        "pandas>=1.1.4",
        "pandas-profiling>=2.9.0",
        "protobuf>=3.13.0",
        "scikit-learn>=0.24.2",
        "scipy>=1.6.2",
    ],  # I get to this in a second
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",  # Again, pick a license
        "Programming Language :: Python :: 3.8",
    ],
)
