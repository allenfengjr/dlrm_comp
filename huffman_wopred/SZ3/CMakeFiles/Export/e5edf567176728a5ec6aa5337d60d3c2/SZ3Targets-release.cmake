#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SZ3::zstd" for configuration "Release"
set_property(TARGET SZ3::zstd APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SZ3::zstd PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libzstd.so"
  IMPORTED_SONAME_RELEASE "libzstd.so"
  )

list(APPEND _cmake_import_check_targets SZ3::zstd )
list(APPEND _cmake_import_check_files_for_SZ3::zstd "${_IMPORT_PREFIX}/lib/libzstd.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
