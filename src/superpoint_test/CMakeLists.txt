cmake_minimum_required(VERSION 3.8)
project(superpoint_test)

# ---------- compile options ----------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++ standard (pick 17 unless you need 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- packages ----------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(mrs_lib REQUIRED)

# ---------- Torch ----------
set(CMAKE_PREFIX_PATH "/opt/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# ---------- target ----------
add_executable(${PROJECT_NAME}
  src/superpoint_detector_node.cpp
  src/SuperPoint.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE ${TORCH_CXX_FLAGS})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  mrs_lib
)

#enforece optimization
#add_compile_options("-O3")

# Helps runtime find torch libs without exporting LD_LIBRARY_PATH every time
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# ---------- install ----------
install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files if you have them
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
