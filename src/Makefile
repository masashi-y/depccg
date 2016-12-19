CC=g++
PYTHON=python2
PROG=myccg
SRCS=main.cpp parser.cpp utils.cpp cat.cpp chainer_tagger.cpp tree.cpp combinator.cpp grammar.cpp cacheable.cpp feat.cpp dep.cpp parser_tools.cpp
OBJS=$(SRCS:.cpp=.o) tagger.o
DEPS=$(SRCS:.cpp=.d)
CFLAGS= -std=c++11 -g -fpic $(OPTIONS) -Wall -march=native
INCLUDES=$(shell $(PYTHON)-config --includes) \
	-I$(shell $(PYTHON) -c 'import numpy; print numpy.get_include()')
LIBDIR=$(shell $(PYTHON)-config --ldflags) -fopenmp

all: $(PROG) $(OBJS)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LIBDIR)

tagger.o: py/tagger.pyx
	python2 setup.py build_ext --inplace
	$(CC) $(CFLAGS) -c py/tagger.c $(INCLUDES) $(LIBDIR)

chainer_tagger.o: chainer_tagger.cpp tagger.o
	$(CC) $(CFLAGS) -c -MMD -MP $< $(INCLUDES) $(LIBDIR)

.cpp.o:
	$(CC) $(CFLAGS) -c -MMD -MP $< -fopenmp

.PHONY: clean
clean:
	rm -rf chainer_tagger.so $(PROG) $(OBJS) $(DEPS) tagger.cpp chainer_tagger.so build *.pyc py/py_utils.so py_utils.c

-include $(DEPS)

