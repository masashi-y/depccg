#! -*- coding: utf-8 -*-

from __future__ import print_function
from Cython.Distutils import build_ext
from setuptools import Extension, setup, find_packages
import numpy
import tempfile
import subprocess
import shutil
import sys
import subprocess
import os
import distutils
import platform
import contextlib


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


COMPILE_OPTIONS = ['-O3', '-Wno-strict-prototypes', '-Wno-unused-function', '-std=c++11']

LINK_OPTIONS = []

CPP_OPTIONS = []

USE_OPENMP_DEFAULT = '0' if sys.platform != 'darwin' else None
if os.environ.get('USE_OPENMP', USE_OPENMP_DEFAULT) == '1':
    COMPILE_OPTIONS.append('-fopenmp')
    LINK_OPTIONS.append('-fopenmp')
    CPP_OPTIONS.append('-fopenmp')

if sys.platform == 'darwin':
    COMPILE_OPTIONS.append('-stdlib=libc++')
    LINK_OPTIONS.append('-lc++')
    # g++ (used by unix compiler on mac) links to libstdc++ as a default lib.
    # See: https://stackoverflow.com/questions/1653047/avoid-linking-to-libstdc
    LINK_OPTIONS.append('-nodefaultlibs')


def generate_cython(root, source):
    print('Cythonizing sources')
    p = subprocess.call([sys.executable,
                         os.path.join(root, 'bin', 'cythonize.py'),
                         source], env=os.environ)
    if p != 0:
        raise RuntimeError('Running cythonize failed')


def generate_cpp(options):
    options = ' '.join(options)
    options = f'OPTIONS={options}'
    with chdir('cpp'):
        p = subprocess.call(["make", options], env=os.environ)
        if p != 0:
            raise RuntimeError('Running cythonize failed')


cpp_sources = ['depccg.cpp',
               'cat.cpp',
               'cacheable.cpp',
               'chart.cpp',
               'combinator.cpp',
               'en_grammar.cpp',
               'feat.cpp',
               'tree.cpp',
               'ja_grammar.cpp',
               'utils.cpp']

pyx_modules = ['depccg.parser',
               'depccg.tree',
               'depccg.cat',
               'depccg.combinator',
               'depccg.utils']


root = os.path.abspath(os.path.dirname(__file__))
generate_cpp(CPP_OPTIONS)
generate_cython(root, 'depccg')


ext_modules = [
        Extension(pyx,
                  [pyx.replace('.', '/') + '.cpp'],
                  include_dirs=['.', numpy.get_include(), 'cpp'],
                  extra_compile_args=COMPILE_OPTIONS,
                  extra_link_args=LINK_OPTIONS +
                  [os.path.join('cpp', cpp.replace('cpp', 'o')) for cpp in cpp_sources],
                  language='c++')
        for pyx in pyx_modules]


setup(
    name="depccg",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    scripts=['bin/depccg_en', 'bin/depccg_ja'],
    zip_safe=False,
)
