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


def check_for_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from yt, thanks to Nathan
    Goldbaum. See https://github.com/pynbody/pynbody/issues/124"""
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    # Get compiler invocation
    compiler = os.environ.get('CC', distutils.sysconfig.get_config_var('CC'))

    # make sure to use just the compiler name without flags
    compiler = compiler.split()[0]

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    filename = 'test.c'
    with open(filename, 'w') as f:
        f.write("""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
    printf(\"Hello from thread %d, nthreads %d\\n\", omp_get_thread_num(), omp_get_num_threads());
}""")

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call([compiler, '-fopenmp', filename],
                                        stdout=fnull, stderr=fnull)
    except OSError:
        exit_code = 1

    # Clean up
    os.chdir(curdir)
    shutil.rmtree(tmpdir)

    if exit_code == 0:
        return True
    else:
        import multiprocessing
        cpus = multiprocessing.cpu_count()
        if cpus > 1:
            print("""WARNING
OpenMP support is not available in your default C compiler, even though
your machine has more than one core available.
Some routines in pynbody are parallelized using OpenMP and these will
only run on one core with your current configuration.
""")
            if platform.uname()[0] == 'Darwin':
                print("""Since you are running on Mac OS, it's likely that the problem here
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
            print("""Continuing your build without OpenMP...\n""")

        return False


def generate_cython(root, source):
    print('Cythonizing sources')
    p = subprocess.call([sys.executable,
                         os.path.join(root, 'bin', 'cythonize.py'),
                         source], env=os.environ)
    if p != 0:
        raise RuntimeError('Running cythonize failed')


cpp_sources = ['depccg.cpp',
               'cat.cpp',
               'chart.cpp',
               'combinator.cpp',
               'en_grammar.cpp',
               'feat.cpp',
               'tree.cpp',
               'ja_grammar.cpp',
               'utils.cpp']

pyx_modules = ['depccg.parser', 'depccg.tree', 'depccg.cat', 'depccg.combinator']

compile_options = "-std=c++11 -O3 -g -fpic -march=native"

# if platform.uname()[0]=='Darwin':
#     compile_options += " -stdlib=libc++"

extra_link_args = ["-fopenmp" if check_for_openmp() else ""]

ext_modules = [
        Extension(pyx.split('.')[0],
                  [pyx.replace('.', '/') + '.cpp'] +
                  [os.path.join('cpp', cpp) for cpp in cpp_sources],
                  include_dirs=['.', numpy.get_include(), 'cpp'],
                  extra_compile_args=compile_options.split(' '),
                  extra_link_args=extra_link_args,
                  language='c++')
        for pyx in pyx_modules]

root = os.path.abspath(os.path.dirname(__file__))
generate_cython(root, 'depccg')

setup(
    name="depccg",
    packages=find_packages(),
    package_data={'': ['*.pyx', '*.pxd', '*.txt', '*.tokens']},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
