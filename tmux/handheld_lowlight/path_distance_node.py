#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import Bool
import math
import time


class PathDistanceNode(Node):
    def __init__(self):
        super().__init__('path_distance_node')

        self.declare_parameter("path_topic", "")
        path_topic = self.get_parameter("path_topic").get_parameter_value().string_value
        #path_topic = self.declare_parameter("path_topic", "").value
        self.get_logger().info(f'PathDistanceNode started, listening to "{path_topic}" topic')
        
        # Create subscription to Path topic
        self.path_subscription = self.create_subscription(
            Path,
            path_topic,
            self.path_callback,
            1
        )
        self.path_subscription  # prevent unused variable warning

        self.path = None

        # Create subscription to /finish topic, which triggers the processing ofthe results
        self.finish_subscription = self.create_subscription(
            Bool,
            '/finish',
            self.finish_callback,
            1
        )
        self.finish_subscription  # prevent unused variable warning

    def finish_callback(self, msg: Bool):
        if not msg.data:
            print("Finish was not succesful, not processing anyting")

        if self.path is None:
            print("Something went wrong, path is empty")
        else:
            if len(self.path.poses) < 2:
                self.get_logger().warn('Path has less than 2 poses, cannot compute distance')
                return
            
            # Get start and end points
            start = self.path.poses[0].pose.position
            end = self.path.poses[-1].pose.position
            
            # Compute Euclidean distance
            distance = math.sqrt(
                (end.x - start.x)**2 +
                (end.y - start.y)**2 +
                (end.z - start.z)**2
            )

            step = 5
            if len(self.path.poses) > step:
                total_length = 0
                for i in range(step+1,len(self.path.poses), step):
                    total_length += math.sqrt(
                        (self.path.poses[i].pose.position.x - self.path.poses[i-step].pose.position.x)**2 +
                        (self.path.poses[i].pose.position.y - self.path.poses[i-step].pose.position.y)**2 +
                        (self.path.poses[i].pose.position.z - self.path.poses[i-step].pose.position.z)**2
                    )

                total_length += math.sqrt(
                    (self.path.poses[-1].pose.position.x - self.path.poses[-step].pose.position.x)**2 +
                    (self.path.poses[-1].pose.position.y - self.path.poses[-step].pose.position.y)**2 +
                    (self.path.poses[-1].pose.position.z - self.path.poses[-step].pose.position.z)**2
                )

                if total_length > 0:
                    self.get_logger().info(
                        f'Endpoints distance: {distance:.2f}m\n'
                        f'Total length: {total_length:.2f}m\n'
                        f'Endpoints distance / Total length: {100*(distance / total_length):.2f}%\n'
                    )
        
    def path_callback(self, msg: Path):
        """Callback function when a Path message is received"""
        self.path = msg

def main(args=None):
    rclpy.init(args=args)
    node = PathDistanceNode()
    rclpy.spin(node)
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
