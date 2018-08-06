cmake_minimum_required(VERSION 3.9)
project(Eyeoc_svm_0)

set(CMAKE_CXX_STANDARD 11)

add_executable(Eyeoc_svm_0 main.cpp util.cpp util.h)

find_package(OPENCV REQUIRED)

target_link_libraries(Eyeoc_svm_0 ${OpenCV_LIBS})