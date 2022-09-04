cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -w -g -fPIC")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

## Find catkin macros and libraries
set(OpenCV_DIR /usr/local/opencv455/lib/cmake/opencv4)
find_package(OpenCV 4.5.5 REQUIRED)
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    image_transport
    sensor_msgs
    cv_bridge
    std_msgs
    rospy
)
# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
# set(Openvino_DIR /home/le/intel/openvino_2022/runtime/cmake)
# find_package(Openvino REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs rospy
  DEPENDS OpenCV 
)

set(Openvino_INCLUDE_DIRS /opt/intel/openvino_2022/runtime/include 
                          /opt/intel/openvino_2022/runtime/include/ie)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Openvino_INCLUDE_DIRS}
  #/opt/intel/openvino_2022/runtime/include/ie
  #${Openvino_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/yololayer.cu)
# target_link_libraries(myplugins nvinfer cudart)

add_executable(sender_yolov5 ${PROJECT_SOURCE_DIR}/src/sender.cpp)

target_link_libraries(sender_yolov5 ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(listener_yolov5 ${PROJECT_SOURCE_DIR}/src/openvino_listener.cpp)

target_link_libraries(listener_yolov5 
                      ${OpenCV_LIBS} 
                      ${catkin_LIBRARIES} 
                      /opt/intel/openvino_2022/runtime/lib/intel64/libopenvino.so
                      /opt/intel/openvino_2022/runtime/3rdparty/tbb/lib/libtbb.so.2)


add_definitions(-O2 -pthread)

