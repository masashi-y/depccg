#! -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

sources = ["parse.pyx",
           "cat.cpp",
           "cat_loader.cpp",
           "chainer_tagger.cpp",
           "chart.cpp",
           "combinator.cpp",
           "dep.cpp",
           "en_grammar.cpp",
           "feat.cpp",
           "ja_grammar.cpp",
           "logger.cpp",
           "parser.cpp",
           "parser_tools.cpp",
           "py/tagger.cpp",
           "tree.cpp",
           "utils.cpp"]

ext_modules = [
        Extension("chainer_tagger",
                  ["py/tagger.pyx"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=["-O3"],
                  language='c++'
                  ),
        Extension("py.py_utils",
                  ["py/py_utils.pyx"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=["-O3", "-ffast-math", "-fPIC"],
                  language='c++'
                  ),
        # Extension("parser",
        #           sources,
        #           include_dirs=[".", numpy.get_include()],
        #           extra_link_args=["-fopenmp"],
        #           extra_compile_args=["-O3", "-std=c++11", "-march=native", "-fpic", "-fopenmp"],
        #           language='c++'
        #           ),
        ]

setup(
        name = "chainer_tagger",
        cmdclass = { "build_ext" : build_ext },
        ext_modules = ext_modules,
)
