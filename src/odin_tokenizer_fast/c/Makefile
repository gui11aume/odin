# src/odin_tokenizer_fast/c/Makefile
#
# Builds the frozen C tokenizer extension.
#
# Targets
# -------
#   all     (default) – compile the .so (tables are committed; run `tables`
#                       to regenerate them from a tokenizer.json)
#   tables  – regenerate odin_tokenizer_tables.h from $(JSON)
#   clean   – remove the .so and the generated header
#
# Called from the project root via:
#   make tokenizer-c          ->  $(MAKE) -C src/odin_tokenizer_fast/c
#   make tokenizer-c-tables   ->  $(MAKE) -C src/odin_tokenizer_fast/c tables
#   make tokenizer-c-clean    ->  $(MAKE) -C src/odin_tokenizer_fast/c clean
#
# uv traverses up the directory tree to find pyproject.toml, so
# "uv run python" works correctly from this subdirectory.

UVPY   := uv run python
CC     := gcc

JSON   ?= /mnt/nvme1/odin_tokenizer/tokenizer.json
TABLES := odin_tokenizer_tables.h

# Resolved lazily so they are only evaluated when actually needed
NUMPY_INC = $(shell $(UVPY) -c "import numpy; print(numpy.get_include())")
PY_INC    = $(shell $(UVPY) -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_EXT    = $(shell $(UVPY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SO_OUT    = ../fast$(PY_EXT)

CFLAGS := -O3 -march=native -funroll-loops -fPIC -shared \
          -I$(NUMPY_INC) -I$(PY_INC) -I. \
          -std=c11 -lm

.PHONY: all tables clean

all:
	@test -f $(TABLES) || { echo "missing $(TABLES) - run 'make tables'"; exit 1; }
	$(CC) $(CFLAGS) -flto odin_tokenizer_fast.c -o $(SO_OUT)
	@echo "Built $(SO_OUT)"

tables: $(TABLES)

$(TABLES): gen_tables.py
	$(UVPY) gen_tables.py $(JSON) $(TABLES)

clean:
	rm -f $(SO_OUT) $(TABLES)
