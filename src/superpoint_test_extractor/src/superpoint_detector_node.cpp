#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <mrs_lib/param_loader.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "SPextractor.h"

namespace superpoint_test
{

class SuperPointDetectorNode : public rclcpp::Node
{
public:
  explicit SuperPointDetectorNode(const rclcpp::NodeOptions & options);

private:
  struct Track
  {
    int id = -1;
    double avg_score = 0.0;
    std::deque<cv::Point2f> points;
  };

  void initialize();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void updateTracks(const std::vector<cv::KeyPoint> & keypoints, const cv::Mat & descriptors);
  void drawTracks(cv::Mat & image) const;
  cv::Scalar colorForTrack(int track_id) const;

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
  double match_threshold_ = 0.7;
  int max_track_length_ = 5;
  int min_track_length_ = 2;
  std::vector<int64_t> img_size_{160, 120};

  std::unique_ptr<ORB_SLAM2::SPextractor> extractor_;

  image_transport::Subscriber sub_image_;
  image_transport::Publisher pub_debug_;

  std::atomic<bool> is_initialized_{false};

  cv::Mat prev_descriptors_;
  std::vector<int> prev_track_ids_;
  std::unordered_map<int, Track> tracks_;
  int next_track_id_ = 0;
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
  this->declare_parameter<double>("match_threshold", 0.7);
  this->declare_parameter<int>("max_track_length", 5);
  this->declare_parameter<int>("min_track_length", 2);
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
  pl.loadParam("match_threshold", match_threshold_);
  pl.loadParam("max_track_length", max_track_length_);
  pl.loadParam("min_track_length", min_track_length_);
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
  RCLCPP_INFO(this->get_logger(), "match_threshold: %.3f", match_threshold_);
  RCLCPP_INFO(this->get_logger(), "max_track_length: %d", max_track_length_);
  RCLCPP_INFO(this->get_logger(), "min_track_length: %d", min_track_length_);
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

    //image resize
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

    updateTracks(kpts, desc);

    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    for (const auto & kp : kpts) {
      cv::circle(vis, kp.pt, 1, cv::Scalar(0, 255, 0), -1);
    }
    drawTracks(vis);

    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding = sensor_msgs::image_encodings::BGR8;
    out.image = vis;
    pub_debug_.publish(out.toImageMsg());

    const auto end_total = std::chrono::high_resolution_clock::now();
    const double extract_time = std::chrono::duration<double, std::milli>(end_extract - start_extract).count();
    const double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    std::size_t active_tracks = 0;
    for (const auto & [id, track] : tracks_) {
      (void)id;
      if (static_cast<int>(track.points.size()) >= min_track_length_) {
        ++active_tracks;
      }
    }
    RCLCPP_INFO(
      this->get_logger(),
      "kpts=%zu desc=%dx%d tracks=%zu extract=%.1f ms total=%.1f ms",
      kpts.size(),
      desc.rows,
      desc.cols,
      active_tracks,
      extract_time,
      total_time);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error: %s", e.what());
  }
}

void SuperPointDetectorNode::updateTracks(const std::vector<cv::KeyPoint> & keypoints, const cv::Mat & descriptors)
{
  std::vector<int> current_track_ids(keypoints.size(), -1);

  if (!prev_descriptors_.empty() && !descriptors.empty()) {
    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(prev_descriptors_, descriptors, matches);

    for (const auto & match : matches) {
      if (match.distance > match_threshold_) {
        continue;
      }
      if (match.queryIdx < 0 || match.queryIdx >= static_cast<int>(prev_track_ids_.size())) {
        continue;
      }
      const int track_id = prev_track_ids_[match.queryIdx];
      if (track_id < 0) {
        continue;
      }

      current_track_ids[match.trainIdx] = track_id;
      auto & track = tracks_[track_id];
      track.id = track_id;
      if (track.points.empty()) {
        track.avg_score = match.distance;
      } else {
        const double length = static_cast<double>(track.points.size());
        track.avg_score = ((length - 1.0) * track.avg_score + match.distance) / length;
      }
      track.points.push_back(keypoints[match.trainIdx].pt);
      while (static_cast<int>(track.points.size()) > max_track_length_) {
        track.points.pop_front();
      }
    }
  }

  for (std::size_t i = 0; i < keypoints.size(); ++i) {
    if (current_track_ids[i] >= 0) {
      continue;
    }
    const int track_id = next_track_id_++;
    current_track_ids[i] = track_id;

    Track track;
    track.id = track_id;
    track.avg_score = 0.0;
    track.points.push_back(keypoints[i].pt);
    tracks_[track_id] = std::move(track);
  }

  for (auto it = tracks_.begin(); it != tracks_.end();) {
    const bool still_visible = std::find(current_track_ids.begin(), current_track_ids.end(), it->first) != current_track_ids.end();
    if (!still_visible) {
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }

  prev_descriptors_ = descriptors.clone();
  prev_track_ids_ = std::move(current_track_ids);
}

void SuperPointDetectorNode::drawTracks(cv::Mat & image) const
{
  for (const auto & [track_id, track] : tracks_) {
    if (static_cast<int>(track.points.size()) < min_track_length_) {
      continue;
    }

    const cv::Scalar color = colorForTrack(track_id);
    for (std::size_t i = 1; i < track.points.size(); ++i) {
      cv::line(image, track.points[i - 1], track.points[i], color, 1, cv::LINE_AA);
    }
    cv::circle(image, track.points.back(), 2, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  }
}

cv::Scalar SuperPointDetectorNode::colorForTrack(int track_id) const
{
  const int hue = (track_id * 37) % 180;
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(color[0], color[1], color[2]);
}

}  // namespace superpoint_test

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<superpoint_test::SuperPointDetectorNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
