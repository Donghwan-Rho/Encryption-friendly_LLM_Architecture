# Detect ccache and use it if possible. Modified from:
# https://crascit.com/2016/04/09/using-ccache-with-cmake/#h-improved-functionality-from-cmake-3-4
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  # Set up wrapper scripts
  set(CCACHE_LAUNCHER_DIR "${CMAKE_CURRENT_LIST_DIR}")
  set(CCACHE_LAUNCHER "${CCACHE_PROGRAM}")

  configure_file(${CCACHE_LAUNCHER_DIR}/launch-cxx.in
                 ${CCACHE_LAUNCHER_DIR}/launch-cxx)
  execute_process(COMMAND chmod a+rx "${CCACHE_LAUNCHER_DIR}/launch-cxx")

  if(CMAKE_GENERATOR STREQUAL "Xcode")
    # Set Xcode project attributes to route compilation and linking through our
    # scripts
    set(CMAKE_XCODE_ATTRIBUTE_CXX "${CCACHE_LAUNCHER_DIR}/launch-cxx")
    set(CMAKE_XCODE_ATTRIBUTE_CXX "${CCACHE_LAUNCHER_DIR}/launch-cxx")
    set(CMAKE_XCODE_ATTRIBUTE_LDPLUSPLUS "${CCACHE_LAUNCHER_DIR}/launch-cxx")
  else()
    # Support Unix Makefiles and Ninja
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_LAUNCHER_DIR}/launch-cxx")
  endif()

  if(CMAKE_CUDA_COMPILER)
    configure_file(${CCACHE_LAUNCHER_DIR}/launch-cuda.in
                   ${CCACHE_LAUNCHER_DIR}/launch-cuda)
    execute_process(COMMAND chmod a+rx "${CCACHE_LAUNCHER_DIR}/launch-cuda")
    if(NOT CMAKE_GENERATOR STREQUAL "Xcode")
      set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_LAUNCHER_DIR}/launch-cuda")
    endif()
  endif()
endif()
