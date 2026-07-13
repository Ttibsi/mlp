CFLAGS = -std=c++23 -g -Wall -Wextra
RAY_LIB_DIR = $(CURDIR)/raylib-6.0_linux_amd64/lib
RAY = -I./raylib-6.0_linux_amd64/include -L$(RAY_LIB_DIR)
RAY_LIBS = -lraylib
RAY_RPATH = -Wl,-rpath,$(RAY_LIB_DIR)
RAY_ENV = LD_LIBRARY_PATH=$(RAY_LIB_DIR):$${LD_LIBRARY_PATH}

.PHONY: all
all:
	mkdir -p bin
	# $(RAY_ENV) g++ $(CFLAGS) examples/xor.cpp -o bin/xor
	# $(RAY_ENV) g++ $(CFLAGS) examples/test.cpp -o bin/test
	$(RAY_ENV) g++ $(CFLAGS) $(RAY) examples/identify.cpp -o bin/identify $(RAY_LIBS) $(RAY_RPATH)

.PHONY: clean
clean:
	rm -rf bin
