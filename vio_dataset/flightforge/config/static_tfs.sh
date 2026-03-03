#!/bin/bash

ros2 run tf2_ros static_transform_publisher \
  --x 0 \
  --y 0 \
  --z 0.0 \
  --roll 0 \
  --pitch 0 \
  --yaw 3.14 \
  --frame-id $UAV_NAME/world_origin \
  --child-frame-id global
