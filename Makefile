CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
LDLIBS   = -lOpenCL -lvulkan

all: bandwidth_test vk_bandwidth

bandwidth_test: bandwidth_test.c
	$(CC) $(CFLAGS) -o $@ $< -lOpenCL

vk_bandwidth: vk_bandwidth.c
	$(CC) $(CFLAGS) -o $@ $< -lvulkan

clean:
	rm -f bandwidth_test vk_bandwidth

.PHONY: all clean
