add_executable(
    hdos_tests
    # test_cholesky.cpp
    # test_triangular_solver.cpp
    test_linear_regression.cpp
    # test_householder.cpp
    # test_householder_left.cpp
    # test_householder_right.cpp
    # test_QR.cpp
    # test_qr_operations.cpp
    # test_jacobi_rotation.cpp
    # test_one_sided_jacobi.cpp
    # test_wide_SVD.cpp
    # test_pca.cpp
    # test_pca_2.cpp
    # test_running_mean.cpp
    # test_running_variance.cpp
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