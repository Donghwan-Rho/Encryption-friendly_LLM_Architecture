cmake_minimum_required(VERSION 3.21)
project(
  HELLM
  VERSION 0.0.1
  DESCRIPTION "Example c++ template project"
  LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(PROJECT_LIB_NAME ${PROJECT_NAME})

# Find HEaaN library
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/external/include)
link_directories(${PROJECT_SOURCE_DIR}/external/lib)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${PROJECT_SOURCE_DIR}/external/lib/cmake)
find_package(HEaaN REQUIRED)
find_package(HEaaN-math REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Warn if we bumped cmake version to 3.23+ but forgot to use
  # CUDA_ARCHITECTURES
  if(${CMAKE_VERSION} GREATER_EQUAL 3.23)
    set(CMAKE_CUDA_ARCHITECTURES "all")
  endif()
  # Note that we include 80-virtual. This should be the latest major CUDA
  # architecture.
  set(CMAKE_CUDA_ARCHITECTURES
      "60-real;61-real;70-real;72-real;75-real;80;86-real;89-real")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

include(BuildType)
include(CTest)
include(CheckLanguage)
include(CCache)

option(BUILD_TEST "Build tests" ON)
option(BUILD_EXAMPLE "Build examples" ON)
option(ENABLE_MULTIGPU "Enable multi-GPU support (requires MPI and NCCL)." ON)

set(SRCS
    src/HEMMer.cpp
    src/LoRA.cpp
    src/MatrixUtils.cpp
    src/Loss.cpp
    src/Exp.cpp
    src/LayerNorm.cpp
    src/ReLU.cpp
    src/Tanh.cpp
    src/Softmax.cpp
    src/TorchTransformerBlock.cpp
    src/TransformerBlock.cpp
    CACHE FILEPATH "Sources" FORCE)

add_library(${PROJECT_LIB_NAME} STATIC ${SRCS})

target_include_directories(
  ${PROJECT_LIB_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

include(CompilerWarnings)
# target_set_warnings(${PROJECT_LIB_NAME})
target_set_warnings_as_errors(${PROJECT_LIB_NAME})

add_library(${PROJECT_LIB_NAME}::${PROJECT_LIB_NAME} ALIAS ${PROJECT_LIB_NAME})

add_library(external-libs INTERFACE)

execute_process(
  COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path, end='')"
  OUTPUT_VARIABLE Torch_DIR)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Torch_DIR})
find_package(Torch REQUIRED)

find_package(OpenMP REQUIRED)

target_include_directories(external-libs INTERFACE include/
                                                   ${TORCH_INCLUDE_DIRS})
target_link_libraries(external-libs INTERFACE torch sentencepiece
                                              OpenMP::OpenMP_CXX nccl)

target_link_libraries(external-libs INTERFACE HEaaN HEaaN-math)

target_link_libraries(external-libs INTERFACE CUDA::cudart_static)

if(ENABLE_MULTIGPU)
  # Check MPI
  find_package(MPI REQUIRED)
  cmake_path(GET MPI_INCLUDE_PATH PARENT_PATH MPI_HOME)
  message(STATUS "Use MPI at ${MPI_HOME}.")
  target_link_libraries(external-libs INTERFACE MPI::MPI_CXX)

  # Check NCCL
  find_package(NCCL REQUIRED)
  message(STATUS "Use NCCL at ${NCCL_HOME}.")
  include_directories(SYSTEM ${NCCL_INCLUDE_PATH})
  target_link_libraries(external-libs INTERFACE ${NCCL_LIB})

  target_compile_definitions(${PROJECT_LIB_NAME} PUBLIC HELLM_MULTIGPU)
endif()

target_link_libraries(${PROJECT_LIB_NAME}
                      PUBLIC $<BUILD_INTERFACE:external-libs>)

# Define exported target
include(GNUInstallDirs)
install(
  TARGETS ${PROJECT_LIB_NAME}
  EXPORT ${PROJECT_LIB_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
  DESTINATION include
  PUBLIC_HEADER DESTINATION include)

# Export the target
install(
  EXPORT ${PROJECT_LIB_NAME}Targets
  FILE ${PROJECT_LIB_NAME}Targets.cmake
  NAMESPACE ${PROJECT_LIB_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME})

# Versioning
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}/Version.hpp" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}/Version.hpp"
        DESTINATION include/${PROJECT_LIB_NAME})

# Install include directory
install(DIRECTORY include/${PROJECT_LIB_NAME} DESTINATION include)

# Make Config.cmake and ConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_LIB_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME})

# Install them
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_LIB_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_LIB_NAME})

if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()
if(BUILD_EXAMPLE)
  add_subdirectory(examples)
endif()
