cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed for mgr-amgf-driver. "
    "Please configure in a separate build directory, for example: "
    "cmake -S . -B external/build/mgr_driver")
endif()

project(mgr_amgf_driver LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS C)
find_package(HYPRE CONFIG QUIET)

if(NOT TARGET HYPRE::HYPRE)
  if(DEFINED HYPRE_INCLUDE_DIRS AND (DEFINED HYPRE_LIBRARIES OR DEFINED HYPRE_LIBRARY))
    if(DEFINED HYPRE_LIBRARIES)
      set(_hypre_libs "${HYPRE_LIBRARIES}")
    else()
      set(_hypre_libs "${HYPRE_LIBRARY}")
    endif()

    add_library(HYPRE::HYPRE INTERFACE IMPORTED)
    set_target_properties(HYPRE::HYPRE PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${_hypre_libs}")
  else()
    message(FATAL_ERROR
      "Could not find HYPRE::HYPRE. Provide a HYPRE CMake package or "
      "set HYPRE_INCLUDE_DIRS and HYPRE_LIBRARIES.")
  endif()
endif()

add_executable(mgr_driver
  mgr_driver_gmres.c
  amgf_mgr_minimal.c
)

target_include_directories(mgr_driver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mgr_driver PRIVATE MPI::MPI_C HYPRE::HYPRE)
