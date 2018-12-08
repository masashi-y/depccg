SRCS=utils.cpp cat.cpp tree.cpp combinator.cpp feat.cpp depccg.cpp chart.cpp cacheable.cpp
OBJS=$(SRCS:.cpp=.o)
DEPS=$(SRCS:.cpp=.d)
CFLAGS=-std=c++11 -g -fpic $(OPTIONS) -Wall -march=native -O3
INCLUDES=

all: $(PROG) $(OBJS)

.cpp.o:
	$(CC) $(CFLAGS) -c -MMD -MP $<

.PHONY: clean
clean:
	rm -rf $(PROG) $(OBJS) $(DEPS)

-include $(DEPS)

