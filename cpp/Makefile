CC=g++
PROG=myccg
SRCS=parser.cpp utils.cpp cat.cpp chainer_tagger.cpp tree.cpp combinator.cpp
OBJS=$(SRCS:.cpp=.o) tagger.o
DEPS=$(SRCS:.cpp=.d)
CFLAGS= -std=c++11 -g -fpic
INCLUDES=$(shell python-config --includes) \
	-I$(shell python -c 'import numpy; print numpy.get_include()')
LIBDIR=$(shell python-config --ldflags)

all: $(PROG) $(OBJS)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LIBDIR)

tagger.o: tagger.pyx
	python setup.py build_ext --inplace
	$(CC) $(CFLAGS) -c tagger.cpp $(INCLUDES) $(LIBDIR)

.cpp.o:
	$(CC) $(CFLAGS) -c -MMD -MP $< $(INCLUDES) $(LIBDIR)

.PHONY: clean
clean:
	rm -rf chainer_tagger.so $(PROG) $(OBJS) $(DEPS) tagger.cpp chainer_tagger.so build

-include $(DEPS)

