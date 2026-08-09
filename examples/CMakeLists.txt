add_executable(
    hdos_linear_regression_example
    linear_regression.cpp
)

target_link_libraries(
    hdos_linear_regression_example
    PRIVATE
        hdos::hdos
)

add_executable(
    hdos_online_moments_example
    online_moments.cpp
)

target_link_libraries(
    hdos_online_moments_example
    PRIVATE
        hdos::hdos
)

add_executable(
    hdos_pca_example
    pca.cpp
)

target_link_libraries(
    hdos_pca_example
    PRIVATE
        hdos::hdos
)