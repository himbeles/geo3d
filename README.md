[![pypi](https://img.shields.io/pypi/v/geo3d.svg)](https://pypi.python.org/pypi/geo3d)
[![Test package](https://github.com/himbeles/geo3d/workflows/Test%20package/badge.svg)](https://github.com/himbeles/geo3d/actions?query=workflow%3A%22Test+package%22)

# geo3d

A python package for performing geometric calculations in 3D.
It allows to 
  - find coordinate system transformations between frames
  - transform points and vectors
  - express points and vectors in different frames
  - create frames from primary and secondary axes vectors
  - align two point groups by minimizing point-to-point distances
  - fit planes to points

Requires Python 3.11 or up.

## Installation

To install the module and its core requirements, run

```sh
pip install geo3d
```

## Development

Maintainers can set up a development environment, including additional requirements for unit testing and documentation:

```sh
uv sync
```

## Usage 
Instructions on basic usage can be found in the jupyter notebook in [`/docs`](./docs), 
which are also deployed to [himbeles.github.io/geo3d](https://himbeles.github.io/geo3d).

## Testing
Unit tests can be run using 
```sh
pytest -s 
```
in package root.

If a coverage report should be generated, run 
```sh
python -m coverage run -m pytest
```

## Building the docs
The documentation can be built from the `.ipynb` documents in the `/docs` folder by running
```sh
make html
```
