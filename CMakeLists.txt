cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(quadruped_drake)

add_subdirectory("towr")

find_package(ifopt 2.0.1 REQUIRED)
find_package(lcm REQUIRED)

add_executable(trunk_mpc trunk_mpc.cpp)
target_link_libraries(trunk_mpc
    PRIVATE 
    towr
    ifopt::ifopt_ipopt
    lcm)

