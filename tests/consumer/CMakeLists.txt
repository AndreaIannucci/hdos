cmake_minimum_required(VERSION 3.20)

project(hdos_consumer LANGUAGES CXX)

find_package(hdos 0.1 CONFIG REQUIRED)

add_executable(
    hdos_consumer
    main.cpp
)

target_link_libraries(
    hdos_consumer
    PRIVATE
        hdos::hdos
)

enable_testing()

add_test(
    NAME hdos_consumer_smoke_test
    COMMAND hdos_consumer
)

