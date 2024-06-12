from distutils.core import setup # Função usada para configurar a construção e instalação de módulos Python.
from Cython.Build import cythonize # Função usada para compilar o código Cython em C.
import numpy # Biblioteca de álgebra linear para Python. numpy.get_include() retorna o diretório onde os arquivos de cabeçalho do NumPy estão localizados.

# Cria o diretório 'monotonic_align' caso ele não exista.
import pathlib

# Cria o diretório 'monotonic_align' caso ele não exista. 
pathlib.Path('monotonic_align').mkdir(exist_ok=True, parents=True)

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
