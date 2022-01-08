#!/usr/bin/env python3
from setuptools import find_packages, setup

requirements = [
    "hydra-core==1.1.1", "pytorch-lightning==0.7.1", "urdfpy==0.0.22", "PyYAML==6.0", "scipy==1.5.4", "pandas==1.1.5",
    "matplotlib==3.3.4", "h5py==3.1.0", "trimesh==3.9.36", "progress==1.6"]

setup(
    name="ancsh",
    version="1.0",
    author="3dlg-hcvc",
    url="https://github.com/3dlg-hcvc/ANCSH-pytorch.git",
    packages=find_packages(exclude=("configs", "tests")),
    include_package_data=True,
    install_requires=requirements,
)
