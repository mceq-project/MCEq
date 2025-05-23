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

# # Detect macOS if(APPLE) # Write out message and version of mac
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch arm64 -arch x86_64")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -framework
# Accelerate -arch arm64 -arch x86_64" ) python_add_library(_libspacc MODULE
# src/MCEq/spacc/spacc.c WITH_SOABI) install(TARGETS _libspacc DESTINATION
# ${INSTALL_DIR}/spacc) endif()

python_add_library(
  _libnrlmsise00 MODULE src/MCEq/geometry/nrlmsise00/nrlmsise-00_data.c
  src/MCEq/geometry/nrlmsise00/nrlmsise-00.c WITH_SOABI)
install(TARGETS _libnrlmsise00 DESTINATION ${INSTALL_DIR}/geometry/nrlmsise00)

python_add_library(_libcorsikaatm MODULE
                   src/MCEq/geometry/corsikaatm/corsikaatm.c WITH_SOABI)
install(TARGETS _libcorsikaatm DESTINATION ${INSTALL_DIR}/geometry/corsikaatm)
