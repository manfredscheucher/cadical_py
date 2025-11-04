cmake_minimum_required(VERSION 3.18)
project(cadical_py LANGUAGES CXX)

# ---- Python & pybind11 ----
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED CONFIG)

# ---- CaDiCaL Pfad finden ----
# Variante A: externe Installation/Checkout per Env-Var
#   export CADICAL=/Users/manfred/github/cadical
set(CADICAL_DIR "$ENV{CADICAL}" CACHE PATH "Path to CaDiCaL checkout (with src/ and build/)")
if (NOT CADICAL_DIR)
  # Variante B: vendored submodule unter extern/cadical
  set(CADICAL_DIR "${CMAKE_SOURCE_DIR}/extern/cadical")
endif()

if (NOT EXISTS "${CADICAL_DIR}/src/cadical.hpp")
  message(FATAL_ERROR "Cannot find cadical.hpp under ${CADICAL_DIR}/src. Set CADICAL env var or add extern/cadical.")
endif()

# Erwartete Artefakte (du hast sie bereits gebaut):
#   ${CADICAL_DIR}/build/libcadical.a
#   ${CADICAL_DIR}/build/ipasir.o
set(CADICAL_LIB "${CADICAL_DIR}/build/libcadical.a")
set(CADICAL_IPASIR_OBJ "${CADICAL_DIR}/build/ipasir.o")

if (NOT EXISTS "${CADICAL_LIB}")
  message(FATAL_ERROR "Missing ${CADICAL_LIB}. Build CaDiCaL first (e.g.,: ./configure && make CXXFLAGS='-O3 -fPIC').")
endif()
if (NOT EXISTS "${CADICAL_IPASIR_OBJ}")
  message(FATAL_ERROR "Missing ${CADICAL_IPASIR_OBJ}. Build CaDiCaL (ipasir.o).")
endif()

# ---- Extension bauen ----
pybind11_add_module(cadical_py MODULE
  src/cadical_py/cadical_py.cpp
)

target_include_directories(cadical_py PRIVATE
  "${CADICAL_DIR}/src"
)

# PIC, C++17, Opt
target_compile_features(cadical_py PRIVATE cxx_std_17)
if (NOT MSVC)
  target_compile_options(cadical_py PRIVATE -O3 -fPIC)
endif()

# Linke statische libcadical + ipasir.o
target_link_libraries(cadical_py PRIVATE
  "${CADICAL_IPASIR_OBJ}"
  "${CADICAL_LIB}"
)

# macOS: Python-Symbole zur Laufzeit auflösen (kein explizites libpython-linking)
if(APPLE)
  target_link_options(cadical_py PRIVATE -undefined dynamic_lookup)
endif()

# Ausgabename exakt "cadical_py" (Suffix/ABI kommt vom Python-Toolchain)
set_target_properties(cadical_py PROPERTIES OUTPUT_NAME "cadical_py")
