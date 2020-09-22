cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(test)

add_subdirectory("towr")

find_package(ifopt 2.0.1 REQUIRED)
find_package(lcm REQUIRED)

add_executable(test test.cpp)
target_link_libraries(test 
    PRIVATE 
    towr
    ifopt::ifopt_ipopt
    lcm)

