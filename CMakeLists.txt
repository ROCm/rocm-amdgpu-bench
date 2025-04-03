cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintended, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(FATAL_ERROR "In-source build")
endif()

# System info
cmake_host_system_information(RESULT LOCALHOST QUERY FQDN)
message(STATUS "Hostname: ${LOCALHOST}")

# Versioning info derived from file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
set(OMNIPERF_FULL_VERSION "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3" ROOFLINE_VERSION
                     "${FULL_VERSION_STRING}")

# Set preferred compiler and supported targets
set(CMAKE_HIP_COMPILER "amdclang++" CACHE STRING "desired c++ compiler" FORCE)
set(CMAKE_HIP_ARCHITECTURES "gfx908;gfx90a;gfx940;gfx941;gfx942;gfx950" CACHE STRING "AMD GPU architectures" FORCE)

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${ROCM_LIB}/cmake/hip ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip /opt/rocm/hip/cmake)

project(
    roofline
    VERSION ${ROOFLINE_VERSION}
    LANGUAGES HIP CXX
    DESCRIPTION "Max compute and bandwidth measurements for roofline plots"
    HOMEPAGE_URL "https://github.com/rocm/rocm-amdgpu-bench")

include(ExternalProject)
include(GNUInstallDirs)

# version control info
find_package(Git)
if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git log --pretty=format:%h -n 1
        OUTPUT_VARIABLE OMNIPERF_GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Git revision: ${OMNIPERF_GIT_REV}")
endif()

set(CMAKE_BUILD_TYPE "Release")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "/opt/rocm"
        CACHE PATH "default install path" FORCE)
endif()
message(STATUS "Installation path: ${CMAKE_INSTALL_PREFIX}")

# Verify CXX Compiler is clang-based
if (CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using ${CMAKE_HIP_COMPILER} to build for amdgpu backend")
else()
  message(FATAL_ERROR "'amdclang++' compiler required to compile for ROCm platform.")
endif()

set(CMAKE_CXX_STANDARD 20)

find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm ${ROCM_DIR})

message(STATUS "Target architecture(s) = ${CMAKE_HIP_ARCHITECTURES}")

set(ROOFLINE_SOURCES
  roofline.cpp
  kernels.cpp
  common.cpp
)

set_source_files_properties(${ROOFLINE_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(roofline ${ROOFLINE_SOURCES})

install(
    TARGETS roofline
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT main)

# ----------
# Packaging
# ----------

message(STATUS "Packaging config...")
set(CPACK_GENERATOR
    "DEB" "RPM"
    CACHE STRING "")
set(CPACK_PACKAGE_NAME
    "${PROJECT_NAME}"
    CACHE STRING "")
set(CPACK_PACKAGE_CONTACT "https://github.com/rocm/rocm-amdgpu-bench")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "rocm-amdgpu-bench: measure max compute and bandwidth.")
set(CPACK_RPM_PACKAGE_DESCRIPTION "rocm-amdgpu-bench provides a rooflining tool
targeting AMD GPUs.")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")

# Package versioning
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}"
    )

# RPM package specific variables
set(CPACK_RPM_PACKAGE_LICENSE "AMD Proprietary")
set(CPACK_RPM_PACKAGE_PROVIDES "${CPACK_PACKAGE_NAME}")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH "top-level package install path")

if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

# Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_LICENSE "AMD Proprietary")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

# Dependencies
set(PACKAGE_REQUIRES
    "rocm-hip-sdk"
    CACHE STRING "Package dependencies")
set(CPACK_RPM_PACKAGE_REQUIRES ${PACKAGE_REQUIRES})
set(CPACK_DEBIAN_PACKAGE_DEPENDS ${PACKAGE_REQUIRES})

# ----- Check for packaging override -----
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.$ENV{ROCM_LIBPATCH_VERSION}")
endif()

if(DEFINED CPACK_RPM_PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE})
else()
    set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()

if(DEFINED CPACK_DEBIAN_PACKAGE_RELEASE)
    set(CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE})
else()
    set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()

# Log package info
message(STATUS "  Package Name: ${CPACK_PACKAGE_NAME}")
message(STATUS "  Package Version: ${CPACK_PACKAGE_VERSION}")
message(STATUS "  RPM Package Release: ${CPACK_RPM_PACKAGE_RELEASE}")
message(STATUS "  Debian Package Release: ${CPACK_DEBIAN_PACKAGE_RELEASE}")
message(STATUS "  Packaging Install Prefix: ${CPACK_PACKAGING_INSTALL_PREFIX}")
message(STATUS "  Package Dependencies: ${PACKAGE_REQUIRES}")
message(STATUS "  CPack Generator: ${CPACK_GENERATOR}")

include(CPack)
