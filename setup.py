#!/usr/bin/env python
from setuptools import setup, Extension, find_packages

if __name__ == "__main__":

    setup(
        name="waffle",
        version="0.0.3",
        author="Ben Shanks",
        author_email="benjamin.shanks@gmail.com",
        packages=find_packages(),
        install_requires=["numpy", "scipy", "pandas", "tables", "future"]
    )
