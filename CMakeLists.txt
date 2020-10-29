cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(quadruped_drake)

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


#add_subdirectory(towr)
include_directories(towr)
link_directories(towr)

find_package(ifopt 2.0.1 REQUIRED)
find_package(lcm REQUIRED)
find_package(towr 1.3.3 REQUIRED)

add_executable(trunk_mpc trunk_mpc.cpp)
target_link_libraries(trunk_mpc
    PUBLIC
    towr::towr
    ifopt::ifopt_ipopt
    lcm)
#target_include_directories(trunk_mpc
#    PRIVATE
#    towr/include)

