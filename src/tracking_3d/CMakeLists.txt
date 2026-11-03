cmake_minimum_required(VERSION 3.0.2)
project(tracking_3d)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  vision_msgs
  visualization_msgs
)
find_package(OpenCV REQUIRED COMPONENTS core)

catkin_package(
  CATKIN_DEPENDS cv_bridge geometry_msgs roscpp sensor_msgs std_msgs tf2 tf2_ros vision_msgs visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(human_3d_tracker src/human_3d_tracker.cpp)
target_link_libraries(human_3d_tracker ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(detection3d_visualizer src/detection3d_visualizer.cpp)
target_link_libraries(detection3d_visualizer ${catkin_LIBRARIES})

install(TARGETS human_3d_tracker detection3d_visualizer
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
