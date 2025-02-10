# Find the NCCL libraries

find_path(
  NCCL_INCLUDE_PATH
  NAMES nccl.h
  PATHS ENV
        CONDA_PREFIX
        ENV
        NCCL_HOME
        ${NCCL_HOME}
        ENV
        NCCL_INCLUDE_PATH
  PATH_SUFFIXES include)

find_library(
  NCCL_LIB
  NAMES nccl
  PATHS ENV
        CONDA_PREFIX
        ENV
        NCCL_HOME
        ${NCCL_HOME}
        ENV
        NCCL_LIB
  PATH_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG)

if(NCCL_FOUND)
  mark_as_advanced(NCCL_INCLUDE_PATH NCCL_LIB)
  if(NOT DEFINED ${NCCL_HOME})
    cmake_path(GET NCCL_INCLUDE_PATH PARENT_PATH NCCL_HOME)
    message(STATUS "set NCCL_HOME: ${NCCL_HOME}")
  endif()
endif()
