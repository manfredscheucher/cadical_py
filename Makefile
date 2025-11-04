# Makefile for cadipy (pybind11 + CaDiCaL ExternalPropagator)

CXX      ?= c++
CXXFLAGS ?= -O3 -std=c++17 -fPIC
LDFLAGS  ?= -shared

# Dein CaDiCaL-Pfad
CADICAL  ?= $(HOME)/github/cadical
INCLUDES := -I"$(CADICAL)/src"
LIBS     := "$(CADICAL)/build/ipasir.o" "$(CADICAL)/build/libcadical.a"

# Aktives Python aus dem (ggf.) aktivierten venv
PYTHON   ?= python3

# Pybind11/Python-Flags dynamisch ermitteln
PY_EXT      := $(shell $(PYTHON) -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')
PY_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
# Auf macOS ist dynamische Auflösung üblich:
PY_LDFLAGS  := -undefined dynamic_lookup

TARGET := cadipy$(PY_EXT)
SRC    := cadipy.cpp

.PHONY: all clean print-flags

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) $(INCLUDES) $(SRC) $(LIBS) $(LDFLAGS) $(PY_LDFLAGS) -o $@

clean:
	rm -f cadipy*.so cadipy*.dylib

print-flags:
	@echo "PYTHON     = $(PYTHON)"
	@echo "PY_EXT     = $(PY_EXT)"
	@echo "PY_INCLUDES= $(PY_INCLUDES)"
	@echo "LDFLAGS    = $(LDFLAGS) $(PY_LDFLAGS)"
