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

if(APPLE)
  # –– declare and name the extension with the correct SOABI suffix
  python_add_library(_libspacc MODULE src/MCEq/spacc/spacc.c WITH_SOABI)

  target_link_libraries(_libspacc PRIVATE "-framework Accelerate")

  # –– install into <INSTALL_DIR>/spacc (so in editable→gemm/spacc)
  install(TARGETS _libspacc LIBRARY # on macOS, MODULE behaves like LIBRARY
                                    DESTINATION ${INSTALL_DIR}/spacc)
endif()
