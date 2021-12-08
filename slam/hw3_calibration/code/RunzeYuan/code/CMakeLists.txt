cmake_minimum_required(VERSION 3.20)
project(calibration)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibration calibration.cpp)
target_link_libraries(calibration ${OpenCV_LIBS})