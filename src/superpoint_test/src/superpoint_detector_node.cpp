#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <mrs_lib/param_loader.h>

// SuperPointSLAM3 extractor
#include "SuperPointExtractor.h"

namespace superpoint_test
{

class SuperPointDetectorNode : public rclcpp::Node
{
public:
  explicit SuperPointDetectorNode(const rclcpp::NodeOptions & options);

private:
  void initialize();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

private:
  // params
  std::string weights_path_;
  std::string image_topic_;
  double threshold_ = 0.015;
  bool do_nms_ = true;     // kept for compatibility, currently unused by this extractor
  bool use_cuda_ = false;  // kept for compatibility, currently unused by this extractor
  std::vector<int64_t> img_size_{160, 120};  // width, height

  int max_features_ = 500;
  double scale_factor_ = 1.2;
  int nlevels_ = 1;

  // extractor
  std::unique_ptr<ORB_SLAM3::SuperPointExtractor> extractor_;

  // image transport
  image_transport::Subscriber sub_image_;
  image_transport::Publisher pub_debug_;

  std::atomic<bool> is_initialized_{false};
};

SuperPointDetectorNode::SuperPointDetectorNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("superpoint_test", options)
{
  initialize();
}

void SuperPointDetectorNode::initialize()
{
  this->declare_parameter<std::string>("config", "");
  this->declare_parameter<std::string>("weights_path", "");
  this->declare_parameter<std::string>("image_topic", "image");
  this->declare_parameter<double>("threshold", 0.015);
  this->declare_parameter<bool>("do_nms", true);
  this->declare_parameter<bool>("use_cuda", false);
  this->declare_parameter<std::vector<int64_t>>("image_size", std::vector<int64_t>{160, 120});

  this->declare_parameter<int>("max_features", 500);
  this->declare_parameter<double>("scale_factor", 1.2);
  this->declare_parameter<int>("nlevels", 1);

  auto node_ptr = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*){});
  mrs_lib::ParamLoader pl(node_ptr);

  pl.addYamlFileFromParam("config");
  pl.loadParam("weights_path", weights_path_);
  pl.loadParam("image_topic", image_topic_);
  pl.loadParam("threshold", threshold_);
  pl.loadParam("do_nms", do_nms_);
  pl.loadParam("use_cuda", use_cuda_);
  pl.loadParam("image_size", img_size_);
  pl.loadParam("max_features", max_features_);
  pl.loadParam("scale_factor", scale_factor_);
  pl.loadParam("nlevels", nlevels_);

  if (!pl.loadedSuccessfully()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load parameters (check config YAML + overrides).");
    rclcpp::shutdown();
    return;
  }

  RCLCPP_INFO(this->get_logger(), "========== SuperPointSLAM3 configuration ==========");
  RCLCPP_INFO(this->get_logger(), "weights_path:  %s", weights_path_.c_str());
  RCLCPP_INFO(this->get_logger(), "image_topic:   %s", image_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "threshold:     %.6f", threshold_);
  RCLCPP_INFO(this->get_logger(), "image_size:    [%ld, %ld]", img_size_[0], img_size_[1]);
  RCLCPP_INFO(this->get_logger(), "max_features:  %d", max_features_);
  RCLCPP_INFO(this->get_logger(), "scale_factor:  %.3f", scale_factor_);
  RCLCPP_INFO(this->get_logger(), "nlevels:       %d", nlevels_);
  RCLCPP_INFO(this->get_logger(), "do_nms:        %s (kept, currently unused)", do_nms_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "use_cuda:      %s", use_cuda_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "===================================================");

  try {
    extractor_ = std::make_unique<ORB_SLAM3::SuperPointExtractor>(
      weights_path_,
      max_features_,
      static_cast<float>(scale_factor_),
      nlevels_,
      use_cuda_,
      static_cast<float>(threshold_));
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to construct SuperPointExtractor: %s", e.what());
    rclcpp::shutdown();
    return;
  }

  image_transport::ImageTransport it(node_ptr);

  sub_image_ = it.subscribe(
    image_topic_, 10,
    std::bind(&SuperPointDetectorNode::onImage, this, std::placeholders::_1));

  pub_debug_ = it.advertise("superpoint/debug", 1);

  RCLCPP_INFO(this->get_logger(), "Initialized, subscribing to: %s", image_topic_.c_str());
  is_initialized_.store(true);
}

void SuperPointDetectorNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  if (!is_initialized_.load()) {
    return;
  }

  const auto start1 = std::chrono::high_resolution_clock::now();
  const std::string color_encoding = "bgr8";

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, color_encoding);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
    return;
  }

  cv::Mat gray;
  if (cv_ptr->image.channels() == 1) {
    gray = cv_ptr->image.clone();
  } else {
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
  }

  // resize
  // if (img_size_.size() == 2 &&
  //     (gray.cols != static_cast<int>(img_size_[0]) || gray.rows != static_cast<int>(img_size_[1]))) {
  //   cv::resize(gray, gray, cv::Size(static_cast<int>(img_size_[0]), static_cast<int>(img_size_[1])));
  // }

  if (gray.type() != CV_8UC1) {
    gray.convertTo(gray, CV_8UC1);
  }

  try {
    const auto start2 = std::chrono::high_resolution_clock::now();

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    std::vector<int> vLappingArea;

    // mask is unused here
    int n_kpts = (*extractor_)(gray, cv::Mat(), kpts, desc, vLappingArea);

    const auto end1 = std::chrono::high_resolution_clock::now();

    // visualization
    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);

    int radius = 1;
    for (const auto & kp : kpts) {
      cv::circle(vis, kp.pt, radius, cv::Scalar(0, 255, 0), -1);
    }

    cv::putText(
      vis,
      "kpts: " + std::to_string(std::max(0, n_kpts)),
      cv::Point(10, 25),
      cv::FONT_HERSHEY_SIMPLEX,
      0.7,
      cv::Scalar(0, 0, 255),
      2);

    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding = color_encoding;
    out.image = vis;

    pub_debug_.publish(out.toImageMsg());

    const auto end2 = std::chrono::high_resolution_clock::now();

    const double pre_time =
      std::chrono::duration<double, std::milli>(start2 - start1).count();
    const double net_time =
      std::chrono::duration<double, std::milli>(end1 - start2).count();
    const double total_time =
      std::chrono::duration<double, std::milli>(end2 - start1).count();

    RCLCPP_INFO(
      get_logger(),
      "kpts=%zu returned=%d desc=%dx%d type=%d pre=%.1f ms forward=%.1f ms total=%.1f ms",
      kpts.size(),
      n_kpts,
      desc.rows,
      desc.cols,
      desc.type(),
      pre_time,
      net_time,
      total_time);

  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "SuperPointExtractor error: %s", e.what());
  }
}

}  // namespace superpoint_test

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<superpoint_test::SuperPointDetectorNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
