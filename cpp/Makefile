SRCS=utils.cpp cat.cpp tree.cpp combinator.cpp feat.cpp depccg.cpp chart.cpp cacheable.cpp
OBJS=$(SRCS:.cpp=.o) morpha.o
DEPS=$(SRCS:.cpp=.d)
CCFLAGS=-std=c++11 -g -fpic $(OPTIONS) -Wall -march=native -O3
CFLAGS= -g -fpic -Wall -march=native -O3
INCLUDES=

all: $(OBJS)

.cpp.o:
	$(CC) $(CCFLAGS) -c -MMD -MP $<

morpha.o: morpha.c
	$(CC) $(CFLAGS) -c -MMD -MP $<

.PHONY: clean
clean:
	rm -rf $(OBJS) $(DEPS)

-include $(DEPS)

