cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(SKBUILD_STATE STREQUAL "editable")
  set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/MCEq) # Specify the
                                                        # installation directory
else()
  set(INSTALL_DIR ${SKBUILD_PROJECT_NAME}) # Specify the installation directory
endif()

python_add_library(
  _libnrlmsise00 MODULE src/MCEq/geometry/nrlmsise00/nrlmsise-00_data.c
  src/MCEq/geometry/nrlmsise00/nrlmsise-00.c WITH_SOABI)
install(TARGETS _libnrlmsise00 DESTINATION ${INSTALL_DIR}/geometry/nrlmsise00)

python_add_library(_libcorsikaatm MODULE
                   src/MCEq/geometry/corsikaatm/corsikaatm.c WITH_SOABI)
install(TARGETS _libcorsikaatm DESTINATION ${INSTALL_DIR}/geometry/corsikaatm)

# Platform-neutral fused ETD2 step stages — no sparse-backend deps, builds on
# every platform. The host backend runs them on every route and precision.
python_add_library(_libetd2_kernels MODULE
                   src/MCEq/etd2_kernels/etd2_kernels.c WITH_SOABI)
# These must produce the same bits as the numpy lowering of the same formula
# table (MCEq.solvers.numerics), which tests/test_solvers.py pins. Contracting
# a * b + c into an FMA is a different rounding of the same association, so it
# would break that wherever FMA is baseline — aarch64, or x86-64 built with
# -march=native. It does not fire on baseline x86-64 only for want of an FMA
# instruction, which is luck, not a guarantee.
target_compile_options(
  _libetd2_kernels PRIVATE
  $<$<C_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>:-ffp-contract=off>)
install(TARGETS _libetd2_kernels DESTINATION ${INSTALL_DIR}/etd2_kernels)

if(APPLE)
  # –– declare and name the extension with the correct SOABI suffix
  python_add_library(_libspacc MODULE src/MCEq/spacc/spacc.c WITH_SOABI)

  # –– install into <INSTALL_DIR>/spacc (so in editable→gemm/spacc)
  install(TARGETS _libspacc LIBRARY # on macOS, MODULE behaves like LIBRARY
                                    DESTINATION ${INSTALL_DIR}/spacc)
endif()
