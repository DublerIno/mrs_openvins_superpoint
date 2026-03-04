cmake_minimum_required(VERSION 3.8)
project(superpoint_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# Point CMake to libtorch - adjust this path as needed
set(CMAKE_PREFIX_PATH "/opt/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# Create the executable - so i can run as ros2 node
add_executable(${PROJECT_NAME}  
src/superpoint_detector_node.cpp
src/SuperPoint.cpp
)
#force C++17
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# Include directories = where to find header files
target_include_directories(${PROJECT_NAME} PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
#link libraries 
target_link_libraries(${PROJECT_NAME} 
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
)

#ros2 things:

ament_target_dependencies(${PROJECT_NAME} 
  cv_bridge
  rclcpp
  sensor_msgs
  image_transport
)

# Helps runtime find torch libs without exporting LD_LIBRARY_PATH every time
set_target_properties(${PROJECT_NAME}  PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Install the executable so ros2 run works
install(TARGETS ${PROJECT_NAME} 
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
