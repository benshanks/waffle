#!/usr/bin/env python

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import sys,os

if __name__ == "__main__":

    setup(
        name="waffle",
        version="0.0.1",
        author="Ben Shanks",
        author_email="benjamin.shanks@gmail.com",
        packages=["waffle"],
        install_requires=["numpy", "scipy", "pandas", "tables", "future"]
    )
