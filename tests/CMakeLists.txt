add_executable(
    hdos_tests
    test_cholesky.cpp
    test_triangular_solver.cpp
)

target_link_libraries(
    hdos_tests
    PRIVATE
        hdos
        GTest::gtest_main
)

# Allows tests to include private detail headers.
target_include_directories(
    hdos_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(hdos_tests)