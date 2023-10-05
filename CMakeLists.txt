cmake_minimum_required(VERSION 3.17)
project(KalmanFilter)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/Eigen)

add_executable(KalmanFilter main.cpp KalmanFilter.h KalmanFilter.h)