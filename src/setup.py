#! -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import tempfile
import subprocess
import shutil
import sys
import os
import distutils

def check_for_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from yt, thanks to Nathan
    Goldbaum. See https://github.com/pynbody/pynbody/issues/124"""
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    # Get compiler invocation
    compiler = os.environ.get('CC',
                              distutils.sysconfig.get_config_var('CC'))

    # make sure to use just the compiler name without flags
    compiler = compiler.split()[0]

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    filename = r'test.c'
    with open(filename,'w') as f :
        f.write(
        "#include <omp.h>\n"
        "#include <stdio.h>\n"
        "int main() {\n"
        "#pragma omp parallel\n"
        "printf(\"Hello from thread %d, nthreads %d\\n\", omp_get_thread_num(), omp_get_num_threads());\n"
        "}"
        )

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call([compiler, '-fopenmp', filename],
                                        stdout=fnull, stderr=fnull)
    except OSError :
        exit_code = 1

    # Clean up
    os.chdir(curdir)
    shutil.rmtree(tmpdir)

    if exit_code == 0:
        return True
    else:
        import multiprocessing, platform
        cpus = multiprocessing.cpu_count()
        if cpus>1:
            print ("""WARNING
OpenMP support is not available in your default C compiler, even though
your machine has more than one core available.
Some routines in pynbody are parallelized using OpenMP and these will
only run on one core with your current configuration.
""")
            if platform.uname()[0]=='Darwin':
                print ("""Since you are running on Mac OS, it's likely that the problem here
is Apple's Clang, which does not support OpenMP at all. The easiest
way to get around this is to download the latest version of gcc from
here: http://hpc.sourceforge.net. After downloading, just point the
CC environment variable to the real gcc and OpenMP support should
get enabled automatically. Something like this -
sudo tar -xzf /path/to/download.tar.gz /
export CC='/usr/local/bin/gcc'
python setup.py clean
python setup.py build
""")
            print ("""Continuing your build without OpenMP...\n""")

        return False


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

compile_options = "-std=c++11 -O3 -g -fpic -march=native"

extra_link_args = ["-fopenmp" if check_for_openmp() else ""]

ext_modules = [
        Extension("depccg",
                  sources,
                  include_dirs=[numpy.get_include(), "."],
                  extra_compile_args=compile_options.split(" "),
                  extra_link_args=extra_link_args,
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
