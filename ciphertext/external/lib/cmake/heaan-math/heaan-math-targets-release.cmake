#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "HEaaN::Math::HEaaN-math" for configuration "Release"
set_property(TARGET HEaaN::Math::HEaaN-math APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HEaaN::Math::HEaaN-math PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libHEaaN-math.so"
  IMPORTED_SONAME_RELEASE "libHEaaN-math.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS HEaaN::Math::HEaaN-math )
list(APPEND _IMPORT_CHECK_FILES_FOR_HEaaN::Math::HEaaN-math "${_IMPORT_PREFIX}/lib/libHEaaN-math.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
