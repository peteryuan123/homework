cmake_minimum_required(VERSION 3.8.3)
project(icp)

set(CMAKE_CXX_FLAGS " -g -Wall ${CMAKE_CXX_FLAGS}")

find_package(PCL REQUIRED)
find_package(Boost REQUIRED filesystem)
find_package(Open3D REQUIRED)

#find_package(Boost REQUIRED COMPONENTS system)
# include_directories(${Open3D_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(icp main.cpp)

target_link_libraries(icp PRIVATE Open3D::Open3D)

# target_link_libraries(colorMulti 
#                         ${OpenCV_LIBS}
#                         ${PCL_LIBRARIES}
#                         ${Boost_LIBRARIES}
# )
