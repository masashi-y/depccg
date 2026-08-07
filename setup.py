from __future__ import annotations

import subprocess
import sys

import numpy
from Cython.Build import build_ext as cython_build_ext
from setuptools import Extension, setup


class BuildExt(cython_build_ext):
    def run(self):
        subprocess.run(["make"], cwd="c", check=True)
        super().run()


compile_options = ["-O3", "-Wall", "-std=c++11"]
link_options = []
if sys.platform == "darwin":
    compile_options.append("-stdlib=libc++")


setup(
    ext_modules=[
        Extension(
            "depccg.morpha",
            ["depccg/morpha.pyx"],
            language="c++",
            extra_compile_args=compile_options,
            extra_link_args=link_options + ["c/morpha.o"],
            include_dirs=[".", "c"],
        ),
        Extension(
            "depccg._parsing",
            ["depccg/parsing.pyx"],
            language="c++",
            extra_compile_args=compile_options,
            extra_link_args=link_options,
            include_dirs=[numpy.get_include(), ".", "depccg"],
        ),
    ],
    cmdclass={"build_ext": BuildExt},
)
