cmake_minimum_required(VERSION 3.20)

project(hdos VERSION 0.1.0 LANGUAGES CXX)

add_library(
    hdos
    src/cholesky.cpp
    src/linear_regression.cpp
    src/online_moments.cpp
    src/pca.cpp
    src/matrix_invert.cpp
    src/matrix_operations.cpp
    src/mat_print.cpp
    src/stable_norm.cpp
    src/triangular_solver.cpp
    src/householder.cpp
    src/QR.cpp
    src/jacobi_rotation.cpp
    src/one_sided_jacobi.cpp
)

target_compile_features(
    hdos
    PUBLIC
        cxx_std_20
)

target_include_directories(
    hdos
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

include(CTest)

if(BUILD_TESTING)
    include(FetchContent)

    FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
)

    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()