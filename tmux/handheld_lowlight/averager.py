#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import pickle


class ImageAverageNode(Node):
    def __init__(self):
        super().__init__('image_average_node')
        
        # Declare and get parameters
        self.declare_parameter('angular_velocity_threshold', 0.5)
        self.declare_parameter('enable_averaging', True)
        self.declare_parameter('buffer_size', 1000)
        
        self.angular_velocity_threshold = self.get_parameter('angular_velocity_threshold').value
        self.enable_averaging = self.get_parameter('enable_averaging').value
        self.buffer_size = self.get_parameter('buffer_size').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Define output paths
        self.ros_dir = os.path.expanduser('~/.ros')
        os.makedirs(self.ros_dir, exist_ok=True)
        self.avg_image_path = os.path.join(self.ros_dir, 'avg_img.png')
        self.state_path = os.path.join(self.ros_dir, 'avg_img_state.pkl')
        
        # Initialize buffer for accumulated images
        self.frame_buffer = []  # List to store recent frames
        self.accumulated_image = None
        self.frame_count = 0
        self.is_grayscale = None
        
        # Load previous state if exists
        self.load_state()
        
        # IMU state
        self.angular_velocity_magnitude = 0.0
        self.is_gathering = False
        
        # Subscribe to the image topic
        self.image_subscription = self.create_subscription(
            Image,
            '/uav1/bluefox/image_raw',
            self.image_callback,
            10
        )
        
        # Subscribe to the IMU topic
        self.imu_subscription = self.create_subscription(
            Imu,
            '/uav1/vio_imu/imu_raw',
            self.imu_callback,
            10
        )
        
        # Publisher for debug/visualization image
        self.debug_publisher = self.create_publisher(
            Image,
            '/uav1/average_image_debug',
            10
        )
        
        # Publisher for corrected (subtracted) image
        self.corrected_publisher = self.create_publisher(
            Image,
            '/uav1/corrected_image',
            10
        )
        
        self.get_logger().info(f'Image Average Node started.')
        self.get_logger().info(f'Angular velocity threshold: {self.angular_velocity_threshold} rad/s')
        self.get_logger().info(f'Averaging enabled: {self.enable_averaging}')
        self.get_logger().info(f'Buffer size: {self.buffer_size} frames')
        self.get_logger().info(f'Average image path: {self.avg_image_path}')
        self.get_logger().info(f'State path: {self.state_path}')
        self.get_logger().info(f'Publishing debug images to: /uav1/average_image_debug')
        self.get_logger().info(f'Publishing corrected images to: /uav1/corrected_image')
        
        if self.frame_count > 0:
            self.get_logger().info(f'Loaded previous state with {self.frame_count} frames')
    
    def load_state(self):
        """Load previous averaging state if it exists."""
        try:
            # Load the average image
            if os.path.exists(self.avg_image_path):
                avg_image = cv2.imread(self.avg_image_path, cv2.IMREAD_UNCHANGED)
                if avg_image is not None:
                    self.get_logger().info(f'Loaded existing average image from {self.avg_image_path}')
                    
                    # Load the state file with buffer and metadata
                    if os.path.exists(self.state_path):
                        with open(self.state_path, 'rb') as f:
                            state = pickle.load(f)
                            self.frame_buffer = state.get('frame_buffer', [])
                            self.frame_count = state.get('frame_count', 0)
                            self.is_grayscale = state.get('is_grayscale', None)
                            
                            # Reconstruct accumulated image from buffer
                            if self.frame_buffer:
                                self.accumulated_image = np.sum(self.frame_buffer, axis=0).astype(np.float64)
                                self.get_logger().info(f'Reconstructed accumulated image from {len(self.frame_buffer)} buffered frames')
                            else:
                                # Fallback: use average * count
                                self.accumulated_image = avg_image.astype(np.float64) * self.frame_count
                                
        except Exception as e:
            self.get_logger().warn(f'Could not load previous state: {str(e)}')
            self.frame_buffer = []
            self.accumulated_image = None
            self.frame_count = 0
    
    def save_state(self):
        """Save current averaging state."""
        try:
            if self.accumulated_image is not None and self.frame_count > 0:
                # Compute and save the average image
                average_image = self.accumulated_image / self.frame_count
                average_image_uint8 = np.clip(average_image, 0, 255).astype(np.uint8)
                cv2.imwrite(self.avg_image_path, average_image_uint8)
                
                # Save the state with buffer
                state = {
                    'frame_buffer': self.frame_buffer,
                    'frame_count': self.frame_count,
                    'is_grayscale': self.is_grayscale
                }
                with open(self.state_path, 'wb') as f:
                    pickle.dump(state, f)
                    
                self.get_logger().info(f'Saved state: {self.frame_count} frames, {len(self.frame_buffer)} in buffer')
                
        except Exception as e:
            self.get_logger().error(f'Error saving state: {str(e)}')
    
    def imu_callback(self, msg):
        """Process IMU data and update angular velocity magnitude."""
        # Calculate magnitude of angular velocity
        omega_x = msg.angular_velocity.x
        omega_y = msg.angular_velocity.y
        omega_z = msg.angular_velocity.z
        
        self.angular_velocity_magnitude = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        
        # Update gathering state
        was_gathering = self.is_gathering
        self.is_gathering = self.angular_velocity_magnitude > self.angular_velocity_threshold
        
        # Log state changes
        if self.is_gathering and not was_gathering:
            self.get_logger().info(f'Started gathering (ω = {self.angular_velocity_magnitude:.3f} rad/s)')
        elif not self.is_gathering and was_gathering:
            self.get_logger().info(f'Stopped gathering (ω = {self.angular_velocity_magnitude:.3f} rad/s)')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Detect if image is grayscale
            if self.is_grayscale is None:
                self.is_grayscale = len(cv_image.shape) == 2 or cv_image.shape[2] == 1
                self.get_logger().info(f'Image type detected: {"Grayscale" if self.is_grayscale else "Color"}')
            
            # Only accumulate if gathering is active AND averaging is enabled
            if self.is_gathering and self.enable_averaging:
                # Convert to float for accumulation (prevents overflow)
                cv_image_float = cv_image.astype(np.float64)
                
                # Initialize accumulated image on first frame
                if self.accumulated_image is None:
                    self.accumulated_image = np.zeros_like(cv_image_float)
                    self.get_logger().info(f'Initialized buffer with shape: {cv_image_float.shape}')
                
                # Add to buffer
                self.frame_buffer.append(cv_image_float.copy())
                
                # If buffer exceeds size, remove oldest frame
                if len(self.frame_buffer) > self.buffer_size:
                    oldest_frame = self.frame_buffer.pop(0)
                    # Remove oldest frame from accumulated image
                    self.accumulated_image -= oldest_frame
                    self.frame_count -= 1
                
                # Add current image to the accumulated buffer
                self.accumulated_image += cv_image_float
                self.frame_count += 1
                
                self.get_logger().info(f'Frame {self.frame_count} accumulated (buffer: {len(self.frame_buffer)}/{self.buffer_size}, ω = {self.angular_velocity_magnitude:.3f} rad/s)')
                
                # Save state after every frame
                self.save_state()
            
            # Compute the average if we have frames
            if self.frame_count > 0:
                average_image = self.accumulated_image / self.frame_count
                average_image_uint8 = np.clip(average_image, 0, 255).astype(np.uint8)
                
                # Compute corrected image: raw - average
                cv_image_float = cv_image.astype(np.float32)
                average_float = average_image.astype(np.float32)
                corrected = cv_image_float - average_float
                
                # Shift and scale to visible range [0, 255]
                # Adding 127 centers the range so negative differences are visible
                corrected_shifted = corrected + 127.0
                corrected_uint8 = np.clip(corrected_shifted, 0, 255).astype(np.uint8)
                
                # Publish corrected image
                if self.is_grayscale:
                    corrected_msg = self.bridge.cv2_to_imgmsg(corrected_uint8, encoding='mono8')
                else:
                    corrected_msg = self.bridge.cv2_to_imgmsg(corrected_uint8, encoding='bgr8')
                corrected_msg.header = msg.header
                self.corrected_publisher.publish(corrected_msg)
                
                # Create debug visualization
                debug_image = self.create_debug_image(average_image_uint8)
                
                # Publish debug image
                if self.is_grayscale:
                    debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                else:
                    debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                debug_msg.header = msg.header
                self.debug_publisher.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def create_debug_image(self, average_image):
        """Create debug visualization with colored frame count overlay."""
        # Convert grayscale to BGR for colored text
        if self.is_grayscale:
            debug_image = cv2.cvtColor(average_image, cv2.COLOR_GRAY2BGR)
        else:
            debug_image = average_image.copy()
        
        # Choose color based on gathering state and averaging enabled
        if self.enable_averaging:
            color = (0, 255, 0) if self.is_gathering else (0, 0, 255)  # Green if gathering, Red if not
        else:
            color = (128, 128, 128)  # Gray if averaging is disabled
        
        # Add text with frame count
        text = f'Frames: {self.frame_count}/{self.buffer_size}'
        if not self.enable_averaging:
            text += ' (READ-ONLY)'
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw black background rectangle for better visibility
        cv2.rectangle(debug_image, (5, 5), (15 + text_width, 15 + text_height + baseline), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(debug_image, text, (10, 10 + text_height), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return debug_image


def main(args=None):
    rclpy.init(args=args)
    
    node = ImageAverageNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Shutting down. Total frames processed: {node.frame_count}')
        node.save_state()  # Save state on shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()