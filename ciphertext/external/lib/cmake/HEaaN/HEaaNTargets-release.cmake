#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "HEaaN::HEaaN" for configuration "Release"
set_property(TARGET HEaaN::HEaaN APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HEaaN::HEaaN PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libHEaaN.so"
  IMPORTED_SONAME_RELEASE "libHEaaN.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS HEaaN::HEaaN )
list(APPEND _IMPORT_CHECK_FILES_FOR_HEaaN::HEaaN "${_IMPORT_PREFIX}/lib/libHEaaN.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
