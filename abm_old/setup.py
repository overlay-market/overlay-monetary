#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

requires = ["mesa"]

setup(
    name="agent_based_model",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requires,
)
