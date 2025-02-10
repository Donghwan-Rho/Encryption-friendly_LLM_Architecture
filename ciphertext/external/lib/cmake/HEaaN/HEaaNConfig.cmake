# ~~~
#
# Copyright (C) 2021-2024 Crypto Lab Inc.
#
# - This file is part of HEaaN homomorphic encryption library.
# - HEaaN cannot be copied and/or distributed without the express permission
#  of Crypto Lab Inc.
#
# ~~~


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was HEaaNConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)


set(USE_OPENMP ON)
set(BUILD_WITH_CUDA ON)

if(USE_OPENMP)
  find_dependency(OpenMP REQUIRED)
endif()

# Only users of the static build require this, so no `REQUIRED`.
if(BUILD_WITH_CUDA)
  find_dependency(CUDAToolkit)
endif()

unset(USE_OPENMP)
unset(BUILD_WITH_CUDA)

include("${CMAKE_CURRENT_LIST_DIR}/HEaaNTargets.cmake")
check_required_components("HEaaN")
