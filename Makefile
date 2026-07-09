CFLAGS = -std=c++23 -g -Wall -Wextra -lm

.PHONY: all
all:
	mkdir -p bin
	# g++ $(CFLAGS) examples/xor.cpp -o bin/xor
	# g++ $(CFLAGS) examples/test.cpp -o bin/test
	g++ $(CFLAGS) examples/identify.cpp -o bin/identify

.PHONY: clean
clean:
	rm -rf bin
