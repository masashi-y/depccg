#! -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


sources = ["depccg.pyx",
          "cat.cpp",
          "cat_loader.cpp",
          "logger.cpp",
          "parser.cpp",
          "parser_tools.cpp",
          "chainer_tagger.cpp",
          "chart.cpp",
          "combinator.cpp",
          "dep.cpp",
          "en_grammar.cpp",
          "feat.cpp",
          "tree.cpp",
          "ja_grammar.cpp",
          "utils.cpp"]

headers = ["cacheable.h",
           "cat.h",
           "cat_loader.h",
           "chainer_tagger.h",
           "chart.h",
           "cmdline.h",
           "combinator.h",
           "configure.h",
           "debug.h",
           "dep.h",
           "feat.h",
           "grammar.h",
           "logger.h",
           "matrix.h",
           "parser.h",
           "parser_tools.h",
           "tree.h",
           "utils.h"]

compile_options = "-Wall -std=c++11 -O3 -g -fpic -march=native"

ext_modules = [
        Extension("depccg",
                  sources,
                  include_dirs=[numpy.get_include(), "."],
                  extra_compile_args=compile_options.split(" "),
                  extra_link_args=["-fopenmp"],
                  language='c++'
                  ),
        # Extension("chainer_tagger",
        #           ["py/tagger.pyx"],
        #           include_dirs=[numpy.get_include()],
        #           extra_compile_args=["-O3"],
        #           language='c++'
        #           ),
        ]

setup(
        name = "depccg",
        cmdclass = { "build_ext" : build_ext },
        ext_modules = ext_modules,
)
