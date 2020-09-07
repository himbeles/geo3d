[![Release on PyPI](https://github.com/himbeles/geo3d/workflows/Publish%20on%20PyPI/badge.svg)](https://pypi.org/project/geo3d/)
[![Test package](https://github.com/himbeles/geo3d/workflows/Test%20package/badge.svg)](https://github.com/himbeles/geo3d/actions?query=workflow%3A%22Test+package%22)

# geo3d

A python package for performing geometric calculations in 3D.
It allows to 
  - find coordinate system transformations between frames
  - transform points and vectors
  - express points and vectors in different frames
  - create frames from primary and secondary axes vectors
  - align two point groups by minimizing point-to-point distances  


## Installation 
To install the module and its core requirements, run
```sh
pip install --user -e .
```
within the base directory. 

To install all requirements, including the ones for unit testing and documentation.

```sh
pip install --user -e .[dev]
```

## Usage 
Instructions on basic usage can be found in the jupyter notebook in `/docs`.


## Testing
Unit tests can be run using 
```sh
pytest -s 
```
in package root. 


## Building the docs
The documentation can be built from the `.ipynb` documents in the `/docs` folder by running
```sh
make html
```
