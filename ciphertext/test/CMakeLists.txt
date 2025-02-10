include(CPM)

cpmaddpackage(
  NAME
  googletest
  GITHUB_REPOSITORY
  google/googletest
  GIT_TAG
  release-1.11.0
  OPTIONS
  "INSTALL_GTEST OFF"
  "BUILD_GMOCK OFF")

add_library(test-base INTERFACE)
target_include_directories(test-base INTERFACE include/)
target_link_libraries(test-base INTERFACE ${PROJECT_LIB_NAME} gtest_main)

add_executable(HEMMerTest HEMMerTest_tmp.cpp)
target_link_libraries(HEMMerTest PRIVATE $<BUILD_INTERFACE:test-base>)
