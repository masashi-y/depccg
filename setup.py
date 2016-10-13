#! -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
        Extension("utils",
                  ["utils.pyx"],
                  ),
        # Extension("astar",
        #           ["astar.pyx"],
        #           language="c++",
        #           libraries=["m"]
        #           ),
        Extension("pqueue",
                  ["pqueue.pyx"],
                  ),
        ]

setup(
        name = "ccg parser",
        cmdclass = { "build_ext" : build_ext },
        ext_modules = ext_modules,
)
