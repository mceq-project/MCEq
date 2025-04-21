cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)


set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect macOS
if(APPLE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch arm64 -arch x86_64")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -framework Accelerate -arch arm64 -arch x86_64")
endif()

# # NRLMSISE-00 library
# add_library(libnrlmsise00 SHARED
# src/MCEq/geometry/nrlmsise00/nrlmsise-00_data.c
# src/MCEq/geometry/nrlmsise00/nrlmsise-00.c
# )
# target_include_directories(libnrlmsise00 PRIVATE src/MCEq/geometry/nrlmsise00)
# set_target_properties(libnrlmsise00 PROPERTIES
# PREFIX ""
# LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/MCEq/geometry/nrlmsise00"
# )
Python_add_library(_libnrlmsise00 MODULE src/MCEq/geometry/nrlmsise00/nrlmsise-00_data.c
src/MCEq/geometry/nrlmsise00/nrlmsise-00.c WITH_SOABI)
install(TARGETS _libnrlmsise00 DESTINATION ${SKBUILD_PROJECT_NAME})

Python_add_library(_libcorsikaatm MODULE src/MCEq/geometry/corsikaatm/corsikaatm.c WITH_SOABI)
install(TARGETS _libcorsikaatm DESTINATION ${SKBUILD_PROJECT_NAME})

# # SPACC library
# add_library(libspacc SHARED
#     src/MCEq/spacc/spacc.c
# )
# set_target_properties(libspacc PROPERTIES
#     PREFIX ""
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/MCEq/spacc"
# )

# # Only build spacc on macOS
# if(NOT APPLE)
#     set_target_properties(libspacc PROPERTIES EXCLUDE_FROM_ALL TRUE)
# endif()