#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

// libtorch
#include <torch/torch.h>

// your SuperPoint wrapper
#include "SuperPoint.h"

// MRS helpers (optional but matches your current style)
#include <mrs_lib/param_loader.h>

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
  bool do_nms_ = true;
  bool use_cuda_ = false;
  std::vector<int64_t> img_size_{160, 120};

  // torch
  torch::Device device_{torch::kCPU};
  std::shared_ptr<ORB_SLAM2::SuperPoint> model_;
  std::unique_ptr<ORB_SLAM2::SPDetector> detector_;

  // image_transport
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
  this->declare_parameter<double>("threshold", 0.015);
  this->declare_parameter<bool>("do_nms", true);
  this->declare_parameter<bool>("use_cuda", false);
  this->declare_parameter<std::vector<int64_t>>("image_size",std::vector<int64_t>{160, 120}); // width, height

  // MRS_lib ParamLoader 
  auto node_ptr = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*){}); // non-owning alias
  mrs_lib::ParamLoader pl(node_ptr);

  pl.addYamlFileFromParam("config");
  pl.loadParam("weights_path", weights_path_);
  pl.loadParam("image_topic", image_topic_);
  pl.loadParam("threshold", threshold_);
  pl.loadParam("do_nms", do_nms_);
  pl.loadParam("use_cuda", use_cuda_);
  pl.loadParam("image_size", img_size_);


  if (!pl.loadedSuccessfully()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load parameters (check `config` YAML + overrides).");
    rclcpp::shutdown();
    return;
  }

  // ---- device ----
  const bool cuda_available = torch::cuda::is_available();
  device_ = torch::kCPU;
  if (use_cuda_ && cuda_available) {
    device_ = torch::kCUDA;
  }

  RCLCPP_INFO(this->get_logger(), "========== SuperPoint configuration ==========");
  RCLCPP_INFO(this->get_logger(), "weights_path: %s", weights_path_.c_str());
  RCLCPP_INFO(this->get_logger(), "image_topic:  %s", image_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "threshold:    %.6f", threshold_);
  RCLCPP_INFO(this->get_logger(), "do_nms:       %s", do_nms_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "use_cuda:     %s", use_cuda_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "CUDA avail:   %s", cuda_available ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "device:       %s", device_.is_cuda() ? "CUDA" : "CPU");
  RCLCPP_INFO(this->get_logger(), "==============================================");

  // ---- model ----
  model_ = std::make_shared<ORB_SLAM2::SuperPoint>();

  try {
    model_->load_weights(weights_path_);
  } catch (const c10::Error & e) {
    RCLCPP_ERROR(this->get_logger(),
                 "torch::load failed for '%s'. Error: %s",
                 weights_path_.c_str(), e.what());
    rclcpp::shutdown();
    return;
  }

  model_->to(device_);
  model_->eval();

  // ---- detector ----
  detector_ = std::make_unique<ORB_SLAM2::SPDetector>(model_);
  detector_->setDevice(device_);

  // ---- image transport ----
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

  //recieved image
  auto cv_ptr = cv_bridge::toCvShare(msg, color_encoding);
  //const cv::Mat& img_bgr = cv_ptr->image;

  //incoming image size
  //RCLCPP_INFO(this->get_logger(),"CV image size: %d x %d", cv_ptr->image.cols, cv_ptr->image.rows);
  
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

  try {
    const auto start2 = std::chrono::high_resolution_clock::now();

    //run forward pass
    detector_->detect(gray);

    const auto end1 = std::chrono::high_resolution_clock::now();

    //get keypoint from propabilities
    std::vector<cv::KeyPoint> kpts;
    detector_->getKeyPoints(static_cast<float>(threshold_), 0, gray.cols, 0, gray.rows, kpts, do_nms_);

    cv::Mat desc;
    detector_->computeDescriptors(kpts, desc);
      
    /*
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "kpts=%zu desc=%dx%d type=%d time=%.1fms img=%dx%d",
                         kpts.size(), desc.rows, desc.cols, desc.type(), ms, gray.cols, gray.rows);
    */

    // visualization
    cv::Mat vis;
    int radius = 1;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    for (const auto & kp : kpts) {
      cv::circle(vis, kp.pt, radius, cv::Scalar(0, 255, 0), -1);
    }

    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding = color_encoding;
    out.image = vis;

    pub_debug_.publish(out.toImageMsg());

    const auto end2 = std::chrono::high_resolution_clock::now();

    const double pre_time = std::chrono::duration<double, std::milli>(start2 - start1).count(); // inital image proccess time
    const double net_time = std::chrono::duration<double, std::milli>(end1 - start2).count(); //sp detect time
    const double total_time = std::chrono::duration<double, std::milli>(end2 - start1).count(); //whole onImage callback
    RCLCPP_INFO(get_logger(), "GetKeypoints.Size -  %zu keypoints.Pre timel %.1f; forward pass time: %.1f ms; total:  %.1f ms", kpts.size(), pre_time, net_time, total_time);


  } catch (const c10::Error & e) {
    RCLCPP_ERROR(this->get_logger(), "SuperPoint torch error: %s", e.what());
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
