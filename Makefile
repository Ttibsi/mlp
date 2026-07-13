CFLAGS = -std=c++23 -g -Wall -Wextra -lm
RAY = -I./raylib-6.0_linux_amd64/include -L./raylib-6.0_linux_amd64/lib

.PHONY: all
all:
	mkdir -p bin
	# g++ $(CFLAGS) examples/xor.cpp -o bin/xor
	# g++ $(CFLAGS) examples/test.cpp -o bin/test
	g++ $(CFLAGS) $(RAY) examples/identify.cpp -o bin/identify

.PHONY: clean
clean:
	rm -rf bin
