#!/usr/bin/env python3
from setuptools import find_packages, setup

requirements = ["hydra-core==1.1.1", "pytorch-lightning==0.7.1"]

setup(
    name="ancsh",
    version="1.0",
    author="3dlg-hcvc",
    url="https://github.com/3dlg-hcvc/ANCSH-pytorch.git",
    packages=find_packages(exclude=("configs", "tests")),
    include_package_data=True,
    install_requires=requirements,
)
