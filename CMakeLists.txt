cmake_minimum_required(VERSION 3.16)
project(vsql_vector_extension)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Extension name
set(EXTENSION_NAME "vsql_vector")

# Find VillageSQL SDK
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# This extension uses storage_api.h which is part of the dev (unstable) ABI.
set(VSQL_USE_DEV_ABI ON)
find_package(VillageSQL REQUIRED)

message(STATUS "VillageSQL SDK found at: ${VillageSQL_PREFIX}")
message(STATUS "VillageSQL include dir: ${VillageSQL_INCLUDE_DIR}")

# Create the vector shared library
add_library(svector SHARED
    src/storage/data_page.cc
    src/storage/root_page.cc
    src/storage/storage.cc
    src/native_vector.cc
    src/vector.cc
)

target_include_directories(svector PRIVATE
    ${VillageSQLExtensionFramework_INCLUDE_DIR}
)

target_compile_options(svector PRIVATE -fPIC)

# On macOS the linker requires all symbols to be resolved at link time by default,
# but vef_storage_* symbols are provided by the server at dlopen time.
if(APPLE)
  target_link_options(svector PRIVATE -undefined dynamic_lookup)
endif()

# Create the VEB package
VEF_CREATE_VEB(
    NAME ${EXTENSION_NAME}
    LIBRARY_TARGET svector
    MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/manifest.json
)

# Page dump diagnostic tool
add_executable(svector_page_dump
    src/storage/tools/svector_page_dump.cc
    src/storage/tools/page_reader.cc
    src/storage/tools/root_page_parser.cc
    src/storage/tools/data_page_parser.cc
)

target_include_directories(svector_page_dump PRIVATE
    ${VillageSQLExtensionFramework_INCLUDE_DIR}
)
