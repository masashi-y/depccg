OBJS= morpha.o
DEPS=$(SRCS:.cpp=.d)
CFLAGS= -g -fpic -Wall -O3
INCLUDES=

all: $(OBJS)


morpha.o: morpha.c
	$(CC) $(CFLAGS) -c -MMD -MP $<

.PHONY: clean
clean:
	rm -rf $(OBJS) $(DEPS)

-include $(DEPS)