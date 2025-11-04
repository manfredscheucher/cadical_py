cmake_minimum_required(VERSION 3.18)
project(cadical_py LANGUAGES CXX)

# -----------------------
# Python & pybind11
# -----------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED CONFIG)

# -----------------------
# Locate CaDiCaL
#   Option A: export CADICAL=/path/to/cadical
#   Option B: git submodule at extern/cadical
# -----------------------
set(CADICAL_DIR "$ENV{CADICAL}" CACHE PATH "Path to CaDiCaL checkout (with src/ and build/)")
if(NOT CADICAL_DIR)
  set(CADICAL_DIR "${CMAKE_SOURCE_DIR}/extern/cadical")
endif()

if(NOT EXISTS "${CADICAL_DIR}/src/cadical.hpp")
  message(FATAL_ERROR
    "Cannot find cadical.hpp under ${CADICAL_DIR}/src.\n"
    "Set the CADICAL environment variable or add extern/cadical.")
endif()

set(CADICAL_LIB        "${CADICAL_DIR}/build/libcadical.a")
set(CADICAL_IPASIR_OBJ "${CADICAL_DIR}/build/ipasir.o")

if(NOT EXISTS "${CADICAL_LIB}")
  message(FATAL_ERROR
    "Missing ${CADICAL_LIB}.\nBuild CaDiCaL first, e.g.:\n"
    "  cd ${CADICAL_DIR} && ./configure && make clean && make CXXFLAGS='-O3 -fPIC'")
endif()
if(NOT EXISTS "${CADICAL_IPASIR_OBJ}")
  message(FATAL_ERROR
    "Missing ${CADICAL_IPASIR_OBJ}.\nBuild CaDiCaL (ipasir.o), e.g.:\n"
    "  cd ${CADICAL_DIR} && ./configure && make clean && make CXXFLAGS='-O3 -fPIC'")
endif()

# -----------------------
# Build extension as cadical_py/_core.*.so
# Source lives at: src/cadical_py/_core.cpp
# -----------------------
pybind11_add_module(cadical_py_core MODULE
  src/cadical_py/_core.cpp
)

# The filename users import from Python is "cadical_py._core"
set_target_properties(cadical_py_core PROPERTIES
  OUTPUT_NAME "_core"
)

target_include_directories(cadical_py_core PRIVATE
  "${CADICAL_DIR}/src"
)

# C++ standard & opts
target_compile_features(cadical_py_core PRIVATE cxx_std_17)
if(NOT MSVC)
  target_compile_options(cadical_py_core PRIVATE -O3 -fPIC)
endif()

# Link CaDiCaL static lib + ipasir.o
# (linking a raw object file is fine)
target_link_libraries(cadical_py_core PRIVATE
  "${CADICAL_IPASIR_OBJ}"
  "${CADICAL_LIB}"
)

# macOS: resolve Python symbols at runtime from the host interpreter
if(APPLE)
  target_link_options(cadical_py_core PRIVATE -undefined dynamic_lookup)
endif()

# Installation path inside the pure-Python package directory
# scikit-build-core will handle package layout; we ensure the extension lands under cadical_py/
install(TARGETS cadical_py_core
  LIBRARY DESTINATION cadical_py           # .so/.dylib on *nix/macOS
  RUNTIME DESTINATION cadical_py           # .pyd on Windows (if ever)
  ARCHIVE DESTINATION cadical_py           # static (not used, but harmless)
)
