# Copyright (c) 2026 VillageSQL Contributors
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.16)
project(vsql_vector_extension)

# Default to an optimized build. An empty CMAKE_BUILD_TYPE yields -O0 (no
# -O flag applied), which builds and runs but leaves the distance kernel
# unoptimized. Override with -DCMAKE_BUILD_TYPE=.
#
# -DWITH_DEBUG=ON means "give me a debug build" -- it takes priority over
# any explicitly passed CMAKE_BUILD_TYPE and forces Debug, so NDEBUG is
# never defined (RelWithDebInfo/Release both define it, which would
# silently compile out every #ifndef NDEBUG debug check in the library).
if(NOT CMAKE_CONFIGURATION_TYPES)
  if(WITH_DEBUG)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
  elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
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
    src/index/hnsw/storage.cc
    src/index/hnsw/graph.cc
    src/index/hnsw/graph_ops.cc
    src/index/hnsw/layer_ops.cc
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

# Distance-kernel codegen (native_vector.h, inlined into vector.cc). -O3 plus
# the reassociation subset lets the compiler split the float reduction into
# SIMD lane-width accumulators and fold multiply-adds, so the L2/cosine/IP/L1
# kernels vectorize (AVX-512/AVX2/NEON per -march). This is the flag set
# pgvector uses; it deliberately omits -ffinite-math-only (implied by full
# -ffast-math) so a float-accumulator overflow still yields a well-ordered
# +inf rather than undefined behaviour. -march is left to the builder
# (pass -DCMAKE_CXX_FLAGS=-march=native); cmake ignores env CXXFLAGS.
target_compile_options(svector PRIVATE
    -O3
    -fassociative-math
    -fno-signed-zeros
    -fno-trapping-math
    -ffp-contract=fast)

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
    src/storage/tools/hnsw_layout.cc
    src/storage/tools/hnsw_graph.cc
)

target_include_directories(svector_page_dump PRIVATE
    ${VillageSQLExtensionFramework_INCLUDE_DIR}
)

# Standalone unit tests. These have no dependency on the VillageSQL SDK.
# They are assert()-based, so NDEBUG (defined by default in Release and
# RelWithDebInfo builds) must never suppress their checks -- without -UNDEBUG
# these binaries would silently skip every assertion and "pass" vacuously.
enable_testing()

add_executable(layer_ops_test
    unittest/layer_ops_test.cc
)

target_compile_options(layer_ops_test PRIVATE -UNDEBUG)

add_test(NAME layer_ops_test COMMAND layer_ops_test)

add_executable(graph_ops_test
    unittest/graph_ops_test.cc
)

target_compile_options(graph_ops_test PRIVATE -UNDEBUG)

add_test(NAME graph_ops_test COMMAND graph_ops_test)

# Storage-layer test. Unlike the header-only graph/layer tests, it drives the
# real column-storage sources against an in-memory fake storage backend, so it
# compiles those sources and needs the SDK include dir (storage_api.h).
add_executable(storage_test
    unittest/storage_test.cc
    src/storage/storage.cc
    src/storage/root_page.cc
    src/storage/data_page.cc
)

target_include_directories(storage_test PRIVATE
    ${VillageSQLExtensionFramework_INCLUDE_DIR}
)

target_compile_options(storage_test PRIVATE -UNDEBUG)

add_test(NAME storage_test COMMAND storage_test)
