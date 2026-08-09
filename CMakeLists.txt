cmake_minimum_required(VERSION 3.20)

project(
    hdos
    VERSION 0.1.0
    DESCRIPTION "High-performance online statistical algorithms"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(HDOS_IS_TOP_LEVEL ON)
else()
    set(HDOS_IS_TOP_LEVEL OFF)
endif()

option(
    HDOS_BUILD_TESTS
    "Build the HDOS test suite"
    ${HDOS_IS_TOP_LEVEL}
)

option(
    HDOS_ENABLE_WARNINGS
    "Enable strict compiler warnings"
    ${HDOS_IS_TOP_LEVEL}
)

option(
    HDOS_BUILD_EXAMPLES
    "Build the HDOS examples"
    OFF
)

add_library(
    hdos
    STATIC
    src/cholesky.cpp
    src/linear_regression.cpp
    src/online_moments.cpp
    src/pca.cpp
    src/matrix_invert.cpp
    src/matrix_operations.cpp
    src/stable_norm.cpp
    src/triangular_solver.cpp
    src/householder.cpp
    src/QR.cpp
    src/jacobi_rotation.cpp
    src/one_sided_jacobi.cpp
)

add_library(hdos::hdos ALIAS hdos)

target_compile_features(
    hdos
    PUBLIC
        cxx_std_20
)

set_target_properties(
    hdos
    PROPERTIES
        CXX_EXTENSIONS OFF
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_include_directories(
    hdos
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(HDOS_ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(
            hdos
            PRIVATE
                /W4
                /permissive-
        )
    else()
        target_compile_options(
            hdos
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wsign-conversion
        )
    endif()
endif()

if(HDOS_BUILD_TESTS)
    enable_testing()

    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY
            https://github.com/google/googletest.git
        GIT_TAG
            v1.17.0
        GIT_SHALLOW
            TRUE
    )

    if(MSVC)
        set(
            gtest_force_shared_crt
            ON
            CACHE BOOL
            ""
            FORCE
        )
    endif()

    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()


if(HDOS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------


set(
    HDOS_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/hdos"
)

install(
    TARGETS hdos
    EXPORT hdosTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
        PATTERN "*.hpp"
)

install(
    EXPORT hdosTargets
    FILE hdosTargets.cmake
    NAMESPACE hdos::
    DESTINATION "${HDOS_INSTALL_CMAKEDIR}"
)


install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    DESTINATION
        "${CMAKE_INSTALL_DATADIR}/hdos"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hdosConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hdosConfig.cmake"
    INSTALL_DESTINATION "${HDOS_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hdosConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/hdosConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/hdosConfigVersion.cmake"
    DESTINATION
        "${HDOS_INSTALL_CMAKEDIR}"
)