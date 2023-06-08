cmake_minimum_required(VERSION 3.5)

project(vpi_orb_realtime)

find_package(vpi 2.2 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} vpi opencv_imgproc opencv_core opencv_imgcodecs opencv_features2d opencv_videoio)
