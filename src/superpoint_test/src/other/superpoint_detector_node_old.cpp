#include <memory>
#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include "image_transport/image_transport.hpp"

/* ROS includes for working with OpenCV and images */
#include <image_transport/camera_subscriber.hpp>

#include <opencv2/imgproc.hpp>

#include "SuperPoint.h"

using namespace std::chrono_literals;

class SuperPointDetectorNode : public rclcpp::Node {
public:
  SuperPointDetectorNode() : Node("superpoint_detector_node") {
    // Params
    weights_path_ = declare_parameter<std::string>("weights_path", "");
    image_topic_  = declare_parameter<std::string>("image_topic", "/camera/image");
    threshold_    = declare_parameter<double>("threshold", 0.015);
    do_nms_       = declare_parameter<bool>("nms", true);
    use_cuda_     = declare_parameter<bool>("use_cuda", false);
    img_size_ = declare_parameter<std::vector<int64_t>>("image_size",std::vector<int64_t>{160, 120}); // width, height


    if (weights_path_.empty()) {
      RCLCPP_ERROR(get_logger(), "Param 'weights_path' is empty. Set it to a libtorch-loadable weight file.");
      throw std::runtime_error("weights_path empty");
    }

    // Create model
    model_ = std::make_shared<ORB_SLAM2::SuperPoint>();

    // Load weights into C++ module
    try {
      model_->load_weights(weights_path_);
    } catch (const c10::Error &e) {
      RCLCPP_ERROR(get_logger(),
                   "torch::load failed for '%s'.\n"
                   "This usually means the file is a Python state_dict (.pth) not directly loadable in C++.\n"
                   "Error: %s",
                   weights_path_.c_str(), e.what());
      throw;
    }
    RCLCPP_INFO(get_logger(), "Weights loaded successfully");


    // Pick device ONCE
    device_ = torch::kCPU;
    bool cuda_available_ = torch::cuda::is_available();
    RCLCPP_INFO(get_logger(), "CUDA available: %s", cuda_available_ ? "true" : "false");
    if (use_cuda_ && torch::cuda::is_available()) {
      device_ = torch::kCUDA;
      RCLCPP_INFO(get_logger(), "Using CUDA");
    } else {
      RCLCPP_INFO(get_logger(), "Using CPU");
    }

    model_->to(device_);
    model_->eval();

    detector_ = std::make_unique<ORB_SLAM2::SPDetector>(model_);

    // Subscribe
    // Subscribe (no image_transport; avoids shared_from_this() in constructor)
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&SuperPointDetectorNode::onImage, this, std::placeholders::_1)
    );

    RCLCPP_INFO(get_logger(), "Subscribed to %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "weights_path=%s threshold=%.4f nms=%s",
                weights_path_.c_str(), threshold_, do_nms_ ? "true" : "false");

    //Publisher 
    dbg_pub_ = this->create_publisher<sensor_msgs::msg::Image>("superpoint/debug", 10);
    RCLCPP_INFO(get_logger(), "Debug publisher created on topic superpoint/debug");
  }

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const std::exception &e) {
      RCLCPP_WARN(get_logger(), "cv_bridge failed: %s", e.what());
      return;
    }

    cv::Mat gray;
    if (cv_ptr->image.channels() == 1) {
      gray = cv_ptr->image;
    } else {
      cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
    }

    // Resize image if needed
    if (img_size_.size() == 2 && (gray.rows != img_size_[1] || gray.cols != img_size_[0])) {
      cv::resize(gray, gray, cv::Size(img_size_[0], img_size_[1]));
    }

    // SPDetector expects CV_8U grayscale.
    if (gray.type() != CV_8UC1) {
      gray.convertTo(gray, CV_8U);
    }

    // IMPORTANT: Your detect() moves model->to(device) every frame currently.
    // If your SPDetector::detect does model->to(device) internally, it's slower but still works.
    // (Recommended: remove model->to(device) from detect() and keep it only here.)
    try {
      const auto start = std::chrono::high_resolution_clock::now();
      detector_->detect(gray, use_cuda_);

      std::vector<cv::KeyPoint> kpts;
      detector_->getKeyPoints(
        static_cast<float>(threshold_),
        0, gray.cols, 0, gray.rows,
        kpts,
        do_nms_
      );
      const auto end = std::chrono::high_resolution_clock::now();

      RCLCPP_INFO(get_logger(), "detect and getkeypoint -  %zu keypoints in %.1f ms",
        kpts.size(),
        std::chrono::duration<double, std::milli>(end - start).count()
      );

      
      cv::Mat desc;

      detector_->computeDescriptors(kpts, desc);

      // Publish debug image with keypoints
      cv::Mat vis;
      cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
      // Draw keypoints
      for (const auto &kp : kpts) {
        cv::circle(vis, kp.pt, 2, cv::Scalar(0,255,0), -1);
      }
      //bridging cv to ros image msg
      auto dbg_msg = cv_bridge::CvImage(msg->header, "bgr8", vis).toImageMsg();
      dbg_pub_->publish(*dbg_msg);

      //info
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "kpts=%zu desc=%dx%d (type=%d) img=%dx%d",
        kpts.size(), desc.rows, desc.cols, desc.type(), gray.cols, gray.rows);

    } catch (const c10::Error &e) {
      RCLCPP_ERROR(get_logger(), "SuperPoint inference error: %s", e.what());
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "Error: %s", e.what());
    }
    
  }

  std::string weights_path_;
  std::string image_topic_;
  double threshold_;
  bool do_nms_;
  bool use_cuda_;
  std::vector<int64_t> img_size_;
  

  torch::Device device_{torch::kCPU};

  std::shared_ptr<ORB_SLAM2::SuperPoint> model_;
  std::unique_ptr<ORB_SLAM2::SPDetector> detector_;

  //subscription to image topic
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  //keypoint publisher
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr dbg_pub_;


};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SuperPointDetectorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
