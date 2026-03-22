#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <mrs_lib/param_loader.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "SPextractor.h"

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
  std::string weights_path_;
  std::string image_topic_;
  int nfeatures_ = 500;
  double scale_factor_ = 1.2;
  int nlevels_ = 4;
  double threshold_ = 0.015;
  double min_threshold_ = 0.007;
  bool do_nms_ = true;
  bool use_cuda_ = false;
  std::vector<int64_t> img_size_{160, 120};

  std::unique_ptr<ORB_SLAM2::SPextractor> extractor_;

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
  // ---- parameters ----
  // declare parameters so they show in `ros2 param list`
  this->declare_parameter<std::string>("config", "");
  this->declare_parameter<std::string>("weights_path", "");
  this->declare_parameter<std::string>("image_topic", "image");
  this->declare_parameter<int>("nfeatures", 500);
  this->declare_parameter<double>("scale_factor", 1.2);
  this->declare_parameter<int>("nlevels", 4);
  this->declare_parameter<double>("threshold", 0.015);
  this->declare_parameter<double>("min_threshold", 0.007);
  this->declare_parameter<bool>("do_nms", true);
  this->declare_parameter<bool>("use_cuda", false);
  this->declare_parameter<std::vector<int64_t>>("image_size", std::vector<int64_t>{160, 120});

  auto node_ptr = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node *) {});
  mrs_lib::ParamLoader pl(node_ptr);

  pl.addYamlFileFromParam("config");
  pl.loadParam("weights_path", weights_path_);
  pl.loadParam("image_topic", image_topic_);
  pl.loadParam("nfeatures", nfeatures_);
  pl.loadParam("scale_factor", scale_factor_);
  pl.loadParam("nlevels", nlevels_);
  pl.loadParam("threshold", threshold_);
  pl.loadParam("min_threshold", min_threshold_);
  pl.loadParam("do_nms", do_nms_);
  pl.loadParam("use_cuda", use_cuda_);
  pl.loadParam("image_size", img_size_);

  if (!pl.loadedSuccessfully()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load parameters (check `config` YAML + overrides).");
    rclcpp::shutdown();
    return;
  }

  RCLCPP_INFO(this->get_logger(), "========== SuperPoint configuration ==========");
  RCLCPP_INFO(this->get_logger(), "weights_path: %s", weights_path_.c_str());
  RCLCPP_INFO(this->get_logger(), "image_topic:  %s", image_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "nfeatures:    %d", nfeatures_);
  RCLCPP_INFO(this->get_logger(), "scale_factor: %.3f", scale_factor_);
  RCLCPP_INFO(this->get_logger(), "nlevels:      %d", nlevels_);
  RCLCPP_INFO(this->get_logger(), "threshold:    %.6f", threshold_);
  RCLCPP_INFO(this->get_logger(), "min_threshold: %.6f", min_threshold_);
  RCLCPP_INFO(this->get_logger(), "do_nms:       %s", do_nms_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "use_cuda:     %s", use_cuda_ ? "true" : "false");
  if (img_size_.size() == 2) {
    RCLCPP_INFO(this->get_logger(), "image_size:   %ld x %ld", img_size_[0], img_size_[1]);
  }
  RCLCPP_INFO(this->get_logger(), "==============================================");

  try {
    extractor_ = std::make_unique<ORB_SLAM2::SPextractor>(
      nfeatures_,
      static_cast<float>(scale_factor_),
      nlevels_,
      static_cast<float>(threshold_),
      static_cast<float>(min_threshold_),
      weights_path_,
      threshold_,
      do_nms_,
      use_cuda_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize SPextractor: %s", e.what());
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
  const auto start_total = std::chrono::high_resolution_clock::now();

  try {
    const auto cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);

    cv::Mat gray;
    if (cv_ptr->image.channels() == 1) {
      gray = cv_ptr->image;
    } else {
      cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
    }

    if (img_size_.size() == 2 &&
        (gray.cols != img_size_[0] || gray.rows != img_size_[1])) {
      cv::resize(gray, gray, cv::Size(static_cast<int>(img_size_[0]), static_cast<int>(img_size_[1])));
    }

    if (gray.type() != CV_8UC1) {
      gray.convertTo(gray, CV_8UC1);
    }

    const auto start_extract = std::chrono::high_resolution_clock::now();
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    (*extractor_)(gray, cv::noArray(), kpts, desc);
    const auto end_extract = std::chrono::high_resolution_clock::now();

    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    for (const auto & kp : kpts) {
      cv::circle(vis, kp.pt, 1, cv::Scalar(0, 255, 0), -1);
    }

    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding = sensor_msgs::image_encodings::BGR8;
    out.image = vis;
    pub_debug_.publish(out.toImageMsg());

    const auto end_total = std::chrono::high_resolution_clock::now();
    const double extract_time = std::chrono::duration<double, std::milli>(end_extract - start_extract).count();
    const double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    RCLCPP_INFO(
      this->get_logger(),
      "kpts=%zu desc=%dx%d extract=%.1f ms total=%.1f ms",
      kpts.size(),
      desc.rows,
      desc.cols,
      extract_time,
      total_time);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error: %s", e.what());
  }
}

}  // namespace superpoint_test

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<superpoint_test::SuperPointDetectorNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
