from distutils.core import setup
from Cython.Build import cythonize
import numpy
import pathlib

pathlib.Path('monotonic_align').mkdir(exist_ok=True, parents=True)

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
