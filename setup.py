from setuptools import Extension, setup, find_packages
import sys
import os
import contextlib
import subprocess


try:
    from Cython.Build import build_ext
except ImportError:
    sys.exit("""\
Could not import Cython, which is required to build depccg extension modules.
Please install cython and numpy prior to installing depccg.\
""")

try:
    import numpy
except ImportError:
    sys.exit("""\
Could not import numpy, which is required to build the extension modules.
Please install cython and numpy prior to installing depccg.\
""")

here = os.path.abspath(os.path.dirname(__file__))


install_requires = [
    line.strip() for line in open(
        os.path.join(here, 'requirements.txt'))
]

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


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


def clean():
    depccg_dir = os.path.join(here, 'depccg')
    for name in os.listdir(depccg_dir):
        file_path = os.path.join(depccg_dir, name)
        if any(file_path.endswith(ext) for ext in ['.so', '.cpp', '.c']):
            os.unlink(file_path)
    with chdir('c'):
        subprocess.call(["make", "clean"], env=os.environ)


def generate_cpp(options):
    options = ' '.join(options)
    options = f'OPTIONS={options}'
    with chdir('c'):
        p = subprocess.call(["make", options], env=os.environ)
        if p != 0:
            raise RuntimeError('Running cythonize failed')


COMPILE_OPTIONS = [
    '-O3',
    '-Wall',
    '-std=c++14'
]

LINK_OPTIONS = []

if sys.platform == 'darwin':
    COMPILE_OPTIONS.append('-stdlib=libc++')
    LINK_OPTIONS.append('-lc++')
    # g++ (used by unix compiler on mac) links to libstdc++ as a default lib.
    # See: https://stackoverflow.com/questions/1653047/avoid-linking-to-libstdc
    LINK_OPTIONS.append('-nodefaultlibs')

ext_modules = [
    Extension(
        'depccg.morpha',
        ['depccg/morpha.pyx'],
        language='c++',
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS + ['c/morpha.o'],
        include_dirs=['.', 'c']
    ),
    Extension(
        'depccg._parsing',
        ['depccg/parsing.pyx'],
        language='c++',
        extra_compile_args=COMPILE_OPTIONS,
        extra_link_args=LINK_OPTIONS,
        include_dirs=[numpy.get_include(), '.', 'depccg'],
    )
]


if len(sys.argv) > 1 and sys.argv[1] == 'clean':
    clean()
else:
    generate_cpp([])

    setup(
        name="depccg",
        version="2.0.1",  # NOQA
        description='A parser for natural language based on combinatory categorial grammar',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Masashi Yoshikawa',
        author_email='yoshikawa@tohoku.ac.jp',
        url='https://github.com/masashi-y/depccg',
        license='MIT License',
        packages=find_packages(),
        package_data={'depccg': ['models/*']},
        scripts=['bin/depccg_en', 'bin/depccg_ja'],
        install_requires=install_requires,
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        zip_safe=False,
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules
    )
