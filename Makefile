.PHONY: all
all:
	g++ -std=c++23 xor.cpp -g -Wall -Wextra -o xor
	g++ -std=c++23 test.cpp -g -Wall -Wextra -o test
	g++ -std=c++23 identify.cpp -g -Wall -Wextra -o identify -lm

.PHONY: clean
clean:
	rm -rf xor test identify
