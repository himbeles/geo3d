from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path
import os

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    required = f.read().splitlines()
with open(path.join(here, "requirements-dev.txt"), encoding="utf-8") as f:
    required_dev = f.read().splitlines()


def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


setup(
    name="geo3d",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="A python package for performing geometric calculations in 3D",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url="https://github.com/himbeles/geo3d",
    # Author details
    author="himbeles",
    author_email="lri@me.com",
    # Choose your license
    license="MIT",
    # What does your project relate to?
    keywords="geometry",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=["contrib", "docs", "tests", "doc"]),
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=required,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": required_dev
        #    'test': ['coverage'],
    },
)
