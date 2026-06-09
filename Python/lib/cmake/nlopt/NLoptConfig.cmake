
# Tell the user project where to find our headers and libraries

set (NLOPT_VERSION "2.11.0")

set (NLOPT_INCLUDE_DIRS "C:/Program Files (x86)/multiRL_Python_Wrapper/include")
set (NLOPT_LIBRARY_DIRS "C:/Program Files (x86)/multiRL_Python_Wrapper/lib")

# Allows loading NLOPT settings from another project
set (NLOPT_CONFIG_FILE "${CMAKE_CURRENT_LIST_FILE}")

# List of compilation flags -DTOTO to export
set (NLOPT_DEFINITIONS "")

# Our library dependencies (contains definitions for IMPORTED targets)
include ("${CMAKE_CURRENT_LIST_DIR}/NLoptLibraryDepends.cmake")

# These are IMPORTED targets created by NLOPTLibraryDepends.cmake
set (NLOPT_LIBRARIES "NLopt::nlopt")

