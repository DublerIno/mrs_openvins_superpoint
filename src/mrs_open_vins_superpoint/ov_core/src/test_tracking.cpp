/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unistd.h>
#include <vector>

#if ROS_AVAILABLE == 1
#include <cv_bridge/cv_bridge.h>
#elif ROS_AVAILABLE == 2
#include <cv_bridge/cv_bridge.hpp>
#endif
#if ROS_AVAILABLE == 1
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#elif ROS_AVAILABLE == 2
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackORB.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

using namespace ov_core;

// Our feature extractor
TrackBase *extractor;

// FPS counter, and other statistics
// https://gamedev.stackexchange.com/a/83174
int frames = 0;
int num_lostfeats = 0;
int num_margfeats = 0;
int featslengths = 0;
int clone_states = 10;
std::deque<double> clonetimes;
double time_start = 0.0;
int proc_width = 0;
int proc_height = 0;
bool enable_dynamic_mask = true;
bool preprocess_light_enable = false;
double preprocess_light_alpha = 1.0;
double preprocess_light_beta = 0.0;
double preprocess_light_gamma = 1.0;
double preprocess_light_flicker_amplitude = 0.0;
int preprocess_light_flicker_period = 0;
uint64_t preprocess_frame_counter = 0;

// How many cameras we will do visual tracking on (mono=1, stereo=2)
int max_cameras = 2;

// Basic tracking logger (file output)
bool tracking_log_enable = true;
std::string tracking_log_path = "test_tracking_stats.log";
int tracking_log_avg_frames = 20;
std::ofstream tracking_log_stream;
int tracking_log_frame_count = 0;
double tracking_log_time_start = 0.0;
double tracking_log_sum_track_feat = 0.0;
double tracking_log_sum_numtracks = 0.0;
double tracking_log_sum_lost_feats = 0.0;
double tracking_log_sum_marg_feats = 0.0;
double tracking_log_sum_lost_track_measurements = 0.0;

static inline void tracking_log_write(const std::string &line) {
  if (!tracking_log_enable || !tracking_log_stream.is_open()) {
    return;
  }
  tracking_log_stream << line << std::endl;
}

static inline double wall_clock_seconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

static inline std::string normalize_topic_name(std::string topic) {
  if (topic.empty() || topic.front() == '/') {
    return topic;
  }
  return "/" + topic;
}

static inline void preprocess_image(const cv::Mat &src, cv::Mat &dst, double image_scale, int max_width) {
  cv::Mat equalized;
  cv::equalizeHist(src, equalized);

  cv::Mat processed = equalized;
  if (preprocess_light_enable) {
    cv::Mat imgf;
    equalized.convertTo(imgf, CV_32FC1, 1.0 / 255.0);

    double beta = preprocess_light_beta;
    if (preprocess_light_flicker_period > 0 && std::abs(preprocess_light_flicker_amplitude) > 1e-12) {
      const double phase = 2.0 * 3.14159265358979323846 *
                           static_cast<double>(preprocess_frame_counter % static_cast<uint64_t>(preprocess_light_flicker_period)) /
                           static_cast<double>(preprocess_light_flicker_period);
      beta += preprocess_light_flicker_amplitude * std::sin(phase);
    }
    preprocess_frame_counter++;

    if (std::abs(preprocess_light_alpha - 1.0) > 1e-12 || std::abs(beta) > 1e-12) {
      imgf = imgf * preprocess_light_alpha + beta;
    }
    cv::min(imgf, 1.0, imgf);
    cv::max(imgf, 0.0, imgf);

    const double gamma = (preprocess_light_gamma > 1e-6) ? preprocess_light_gamma : 1.0;
    if (std::abs(gamma - 1.0) > 1e-12) {
      cv::pow(imgf, gamma, imgf);
    }
    imgf.convertTo(processed, CV_8UC1, 255.0);
  }

  double effective_scale = image_scale;
  if (effective_scale <= 0.0) {
    effective_scale = 1.0;
  }
  if (max_width > 0 && processed.cols > max_width) {
    const double width_scale = static_cast<double>(max_width) / static_cast<double>(processed.cols);
    effective_scale = std::min(effective_scale, width_scale);
  }

  if (std::abs(effective_scale - 1.0) < 1e-6) {
    dst = processed;
    return;
  }
  cv::resize(processed, dst, cv::Size(), effective_scale, effective_scale, cv::INTER_AREA);
}

static inline bool ends_with_case_insensitive(const std::string &value, const std::string &suffix) {
  if (value.size() < suffix.size()) {
    return false;
  }
  for (size_t i = 0; i < suffix.size(); i++) {
    const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(value[value.size() - suffix.size() + i])));
    const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[i])));
    if (a != b) {
      return false;
    }
  }
  return true;
}

// Our master function for tracking
void handle_stereo(double time0, double time1, cv::Mat img0, cv::Mat img1);

// Main function
int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path.txt";
  if (argc > 1) {
    config_path = argv[1];
  }

  // Initialize this as a ROS node
#if ROS_AVAILABLE == 1
  ros::init(argc, argv, "test_tracking");
  auto nh = std::make_shared<ros::NodeHandle>("~");
  nh->param<std::string>("config_path", config_path, config_path);
#elif ROS_AVAILABLE == 2
  rclcpp::init(argc, argv);
  auto options = rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("test_tracking", options);
  if (!node->has_parameter("config_path")) {
    node->declare_parameter<std::string>("config_path", config_path);
  }
  node->get_parameter("config_path", config_path);
#endif

  // Load parameters
  auto parser = std::make_shared<ov_core::YamlParser>(config_path, false);
#if ROS_AVAILABLE == 1
  parser->set_node_handler(nh);
#elif ROS_AVAILABLE == 2
  parser->set_node(node);
#endif

  // Verbosity
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Our camera topics (left and right stereo)
  std::string topic_camera0 = "/cam0/image_raw";
  std::string topic_camera1 = "/cam1/image_raw";
#if ROS_AVAILABLE == 1
  nh->param<std::string>("topic_camera0", topic_camera0, "/cam0/image_raw");
  nh->param<std::string>("topic_camera1", topic_camera1, "/cam1/image_raw");
#elif ROS_AVAILABLE == 2
  if (!node->has_parameter("topic_camera0")) {
    node->declare_parameter<std::string>("topic_camera0", topic_camera0);
  }
  if (!node->has_parameter("topic_camera1")) {
    node->declare_parameter<std::string>("topic_camera1", topic_camera1);
  }
  node->get_parameter("topic_camera0", topic_camera0);
  node->get_parameter("topic_camera1", topic_camera1);
#endif
  parser->parse_external("relative_config_imucam", "cam" + std::to_string(0), "rostopic", topic_camera0);
  parser->parse_external("relative_config_imucam", "cam" + std::to_string(1), "rostopic", topic_camera1);
  topic_camera0 = normalize_topic_name(topic_camera0);
  topic_camera1 = normalize_topic_name(topic_camera1);
  const std::string topic_camera0_compressed =
      ends_with_case_insensitive(topic_camera0, "/compressed") ? topic_camera0 : topic_camera0 + "/compressed";
  const std::string topic_camera1_compressed =
      ends_with_case_insensitive(topic_camera1, "/compressed") ? topic_camera1 : topic_camera1 + "/compressed";

  // Location of the ROS bag we want to read in
  std::string path_to_bag = "/home/patrick/datasets/euroc_mav/V1_01_easy.bag";
  std::string bag_storage_id = "auto";
#if ROS_AVAILABLE == 1
  nh->param<std::string>("path_bag", path_to_bag, "/home/patrick/datasets/euroc_mav/V1_01_easy.bag");
#elif ROS_AVAILABLE == 2
  if (!node->has_parameter("path_bag")) {
    node->declare_parameter<std::string>("path_bag", path_to_bag);
  }
  if (!node->has_parameter("bag_storage_id")) {
    node->declare_parameter<std::string>("bag_storage_id", bag_storage_id);
  }
  node->get_parameter("path_bag", path_to_bag);
  node->get_parameter("bag_storage_id", bag_storage_id);
#endif
  // nh->param<std::string>("path_bag", path_to_bag, "/home/patrick/datasets/rpng_aruco/aruco_room_01.bag");
  PRINT_INFO("ros bag path is: %s\n", path_to_bag.c_str());
  PRINT_INFO("ros bag storage id is: %s\n", bag_storage_id.c_str());

  // Get our start location and how much of the bag we want to play
  // Make the bag duration < 0 to just process to the end of the bag
  double bag_start = 0.0;
  double bag_durr = -1.0;
  double image_scale = 1.0;
  int max_width = 0;
#if ROS_AVAILABLE == 1
  nh->param<double>("bag_start", bag_start, 0);
  nh->param<double>("bag_durr", bag_durr, -1);
#elif ROS_AVAILABLE == 2
  if (!node->has_parameter("bag_start")) {
    node->declare_parameter<double>("bag_start", bag_start);
  }
  if (!node->has_parameter("bag_durr")) {
    node->declare_parameter<double>("bag_durr", bag_durr);
  }
  if (!node->has_parameter("image_scale")) {
    node->declare_parameter<double>("image_scale", image_scale);
  }
  if (!node->has_parameter("max_width")) {
    node->declare_parameter<int>("max_width", max_width);
  }
  node->get_parameter("bag_start", bag_start);
  node->get_parameter("bag_durr", bag_durr);
  node->get_parameter("image_scale", image_scale);
  node->get_parameter("max_width", max_width);
#endif
  parser->parse_config("image_scale", image_scale, false);
  parser->parse_config("max_width", max_width, false);
  parser->parse_config("enable_dynamic_mask", enable_dynamic_mask, false);
  parser->parse_config("preprocess_light_enable", preprocess_light_enable, false);
  parser->parse_config("preprocess_light_alpha", preprocess_light_alpha, false);
  parser->parse_config("preprocess_light_beta", preprocess_light_beta, false);
  parser->parse_config("preprocess_light_gamma", preprocess_light_gamma, false);
  parser->parse_config("preprocess_light_flicker_amplitude", preprocess_light_flicker_amplitude, false);
  parser->parse_config("preprocess_light_flicker_period", preprocess_light_flicker_period, false);
  parser->parse_config("tracking_log_enable", tracking_log_enable, false);
  parser->parse_config("tracking_log_path", tracking_log_path, false);
  parser->parse_config("tracking_log_avg_frames", tracking_log_avg_frames, false);
  PRINT_INFO("tracking image scale: %.3f (max_width=%d)\n", image_scale, max_width);
  PRINT_INFO("enable dynamic mask: %d\n", enable_dynamic_mask);
  PRINT_INFO("preprocess light: enable=%d alpha=%.3f beta=%.3f gamma=%.3f flicker_amp=%.3f flicker_period=%d\n",
             (int)preprocess_light_enable, preprocess_light_alpha, preprocess_light_beta, preprocess_light_gamma,
             preprocess_light_flicker_amplitude, preprocess_light_flicker_period);
  PRINT_INFO("tracking log: enable=%d path=%s avg_frames=%d\n", (int)tracking_log_enable, tracking_log_path.c_str(), tracking_log_avg_frames);

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(4);

  // Parameters for our extractor
  int num_pts = 200;
  int num_aruco = 1024;
  int fast_threshold = 20;
  int grid_x = 5;
  int grid_y = 3;
  int min_px_dist = 10;
  double knn_ratio = 0.70;
  bool do_downsizing = false;
  bool use_stereo = false;
  std::string tracker_type = "KLT";

  // TrackDescriptor / SuperPoint params
  std::string sp_weights_path =
      "/home/sponer/ws_openvins_superpoint/src/mrs_open_vins_superpoint/ov_core/src/track/superpoint_model_weights.bin";
  double sp_threshold = 0.015;
  bool sp_do_nms = true;
  bool sp_use_cuda = true;
  int sp_nfeatures = 500;
  float sp_scaleFactor = 1.2f;
  int sp_nlevels = 4;
  float sp_iniThFAST = 0.015f;
  float sp_minThFAST = 0.007f;

  parser->parse_config("max_cameras", max_cameras, false);
  parser->parse_config("num_pts", num_pts, false);
  parser->parse_config("num_aruco", num_aruco, false);
  parser->parse_config("clone_states", clone_states, false);
  parser->parse_config("fast_threshold", fast_threshold, false);
  parser->parse_config("grid_x", grid_x, false);
  parser->parse_config("grid_y", grid_y, false);
  parser->parse_config("min_px_dist", min_px_dist, false);
  parser->parse_config("knn_ratio", knn_ratio, false);
  parser->parse_config("do_downsizing", do_downsizing, false);
  parser->parse_config("use_stereo", use_stereo, false);
  parser->parse_config("tracker_type", tracker_type, false);
  parser->parse_config("weights_path", sp_weights_path, false);
  parser->parse_config("sp_threshold", sp_threshold, false);
  parser->parse_config("do_nms", sp_do_nms, false);
  parser->parse_config("use_cuda", sp_use_cuda, false);
  parser->parse_config("sp_nfeatures", sp_nfeatures, false);
  parser->parse_config("sp_scaleFactor", sp_scaleFactor, false);
  parser->parse_config("sp_nlevels", sp_nlevels, false);
  parser->parse_config("sp_iniThFAST", sp_iniThFAST, false);
  parser->parse_config("sp_minThFAST", sp_minThFAST, false);

  // Histogram method
  ov_core::TrackBase::HistogramMethod method;
  std::string histogram_method_str = "HISTOGRAM";
  parser->parse_config("histogram_method", histogram_method_str, false);
  if (histogram_method_str == "NONE") {
    method = ov_core::TrackBase::NONE;
  } else if (histogram_method_str == "HISTOGRAM") {
    method = ov_core::TrackBase::HISTOGRAM;
  } else if (histogram_method_str == "CLAHE") {
    method = ov_core::TrackBase::CLAHE;
  } else {
    printf(RED "invalid feature histogram specified:\n" RESET);
    printf(RED "\t- NONE\n" RESET);
    printf(RED "\t- HISTOGRAM\n" RESET);
    printf(RED "\t- CLAHE\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Debug print!
  PRINT_DEBUG("max cameras: %d\n", max_cameras);
  PRINT_DEBUG("max features: %d\n", num_pts);
  PRINT_DEBUG("max aruco: %d\n", num_aruco);
  PRINT_DEBUG("clone states: %d\n", clone_states);
  PRINT_DEBUG("grid size: %d x %d\n", grid_x, grid_y);
  PRINT_DEBUG("fast threshold: %d\n", fast_threshold);
  PRINT_DEBUG("min pixel distance: %d\n", min_px_dist);
  PRINT_DEBUG("downsize aruco image: %d\n", do_downsizing);
  PRINT_DEBUG("stereo tracking: %d\n", use_stereo);
  PRINT_DEBUG("tracker type: %s\n", tracker_type.c_str());

  // Fake camera info (we don't need this, as we are not using the normalized coordinates for anything)
  std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras;
  for (int i = 0; i < 2; i++) {
    Eigen::Matrix<double, 8, 1> cam0_calib;
    cam0_calib << 1, 1, 0, 0, 0, 0, 0, 0;
    std::shared_ptr<CamBase> camera_calib = std::make_shared<CamRadtan>(100, 100);
    camera_calib->set_value(cam0_calib);
    cameras.insert({i, camera_calib});
  }

  // Lets make a feature extractor selected at runtime
  std::string tracker_type_upper = tracker_type;
  std::transform(tracker_type_upper.begin(), tracker_type_upper.end(), tracker_type_upper.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

  if (tracker_type_upper == "KLT") {
    extractor = new TrackKLT(cameras, num_pts, num_aruco, use_stereo, method, fast_threshold, grid_x, grid_y, min_px_dist);
  } else if (tracker_type_upper == "ORB" || tracker_type_upper == "TRACKORB") {
    extractor = new TrackORB(cameras, num_pts, num_aruco, use_stereo, method, fast_threshold, grid_x, grid_y, min_px_dist, knn_ratio);
  } else if (tracker_type_upper == "DESCRIPTOR" || tracker_type_upper == "TRACKDESCRIPTOR") {
    extractor = new TrackDescriptor(cameras, num_pts, num_aruco, use_stereo, method, fast_threshold, grid_x, grid_y, min_px_dist,
                                    knn_ratio, sp_weights_path, sp_threshold, sp_do_nms, sp_use_cuda, sp_nfeatures, sp_scaleFactor,
                                    sp_nlevels, sp_iniThFAST, sp_minThFAST);
  } else if (tracker_type_upper == "ARUCO" || tracker_type_upper == "TRACKARUCO") {
    extractor = new TrackAruco(cameras, num_aruco, use_stereo, method, do_downsizing);
  } else {
    PRINT_ERROR(RED "invalid tracker_type specified: %s\n" RESET, tracker_type.c_str());
    PRINT_ERROR(RED "\t- KLT\n" RESET);
    PRINT_ERROR(RED "\t- ORB\n" RESET);
    PRINT_ERROR(RED "\t- DESCRIPTOR\n" RESET);
    PRINT_ERROR(RED "\t- ARUCO\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  if (tracking_log_avg_frames < 1) {
    tracking_log_avg_frames = 1;
  }
  if (tracking_log_enable) {
    tracking_log_stream.open(tracking_log_path, std::ofstream::out | std::ofstream::trunc);
    if (!tracking_log_stream.is_open()) {
      PRINT_WARNING(YELLOW "failed to open tracking log file: %s\n" RESET, tracking_log_path.c_str());
      tracking_log_enable = false;
    } else {
      tracking_log_stream << std::fixed << std::setprecision(6);
      tracking_log_write("# test_tracking basic log");
      tracking_log_write("# parsed_params");
      tracking_log_write("config_path=" + config_path);
      tracking_log_write("bag_path=" + path_to_bag);
      tracking_log_write("bag_storage_id=" + bag_storage_id);
      tracking_log_write("topic_camera0=" + topic_camera0);
      tracking_log_write("topic_camera1=" + topic_camera1);
      tracking_log_write("tracker_type_raw=" + tracker_type);
      tracking_log_write("tracker_type_chosen=" + tracker_type_upper);
      tracking_log_write("max_cameras=" + std::to_string(max_cameras));
      tracking_log_write("use_stereo=" + std::to_string((int)use_stereo));
      tracking_log_write("num_pts=" + std::to_string(num_pts));
      tracking_log_write("num_aruco=" + std::to_string(num_aruco));
      tracking_log_write("clone_states=" + std::to_string(clone_states));
      tracking_log_write("fast_threshold=" + std::to_string(fast_threshold));
      tracking_log_write("grid_x=" + std::to_string(grid_x));
      tracking_log_write("grid_y=" + std::to_string(grid_y));
      tracking_log_write("min_px_dist=" + std::to_string(min_px_dist));
      tracking_log_write("knn_ratio=" + std::to_string(knn_ratio));
      tracking_log_write("weights_path=" + sp_weights_path);
      tracking_log_write("sp_threshold=" + std::to_string(sp_threshold));
      tracking_log_write("do_nms=" + std::to_string((int)sp_do_nms));
      tracking_log_write("use_cuda=" + std::to_string((int)sp_use_cuda));
      tracking_log_write("sp_nfeatures=" + std::to_string(sp_nfeatures));
      tracking_log_write("tracking_log_avg_frames=" + std::to_string(tracking_log_avg_frames));
      tracking_log_write("# avg_window columns: frame_count,fps,track_feat,numtracks,lost_feats_per_frame,marg_tracks_per_frame,avg_track_length_lost_feat");
    }
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Record the start time for our FPS counter
  time_start = wall_clock_seconds();

  // Our stereo pair we have
  bool has_left = false;
  bool has_right = false;
  bool processed_any = false;
  cv::Mat img0, img1;
  double time0 = 0.0;
  double time1 = 0.0;

#if ROS_AVAILABLE == 1
  // Load rosbag here, and find messages we can play
  rosbag::Bag bag;
  bag.open(path_to_bag, rosbag::bagmode::Read);

  // We should load the bag as a view
  // Here we go from beginning of the bag to the end of the bag
  rosbag::View view_full;
  rosbag::View view;

  // Start a few seconds in from the full view time
  // If we have a negative duration then use the full bag length
  view_full.addQuery(bag);
  ros::Time time_init = view_full.getBeginTime();
  time_init += ros::Duration(bag_start);
  ros::Time time_finish = (bag_durr < 0) ? view_full.getEndTime() : time_init + ros::Duration(bag_durr);
  PRINT_DEBUG("time start = %.6f\n", time_init.toSec());
  PRINT_DEBUG("time end   = %.6f\n", time_finish.toSec());
  view.addQuery(bag, time_init, time_finish);

  // Check to make sure we have data to play
  if (view.size() == 0) {
    PRINT_ERROR(RED "No messages to play on specified topics. Exiting.\n" RESET);
    ros::shutdown();
    return EXIT_FAILURE;
  }

  time0 = time_init.toSec();
  time1 = time_init.toSec();

  // Step through the rosbag
  for (const rosbag::MessageInstance &m : view) {

    // If ros is wants us to stop, break out
    if (!ros::ok())
      break;

    const std::string msg_topic = normalize_topic_name(m.getTopic());

    // Handle LEFT camera (raw image)
    sensor_msgs::Image::ConstPtr s0 = m.instantiate<sensor_msgs::Image>();
    if (s0 != nullptr && msg_topic == topic_camera0) {
      // Get the image
      cv_bridge::CvImageConstPtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvShare(s0, sensor_msgs::image_encodings::MONO8);
      } catch (cv_bridge::Exception &e) {
        PRINT_ERROR(RED "cv_bridge exception: %s\n" RESET, e.what());
        continue;
      }
      // Save to our temp variable
      has_left = true;
      preprocess_image(cv_ptr->image, img0, image_scale, max_width);
      time0 = cv_ptr->header.stamp.toSec();
    }
    // Handle LEFT camera (compressed image)
    sensor_msgs::CompressedImage::ConstPtr c0 = m.instantiate<sensor_msgs::CompressedImage>();
    if (c0 != nullptr && msg_topic == topic_camera0_compressed) {
      cv_bridge::CvImagePtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvCopy(c0, sensor_msgs::image_encodings::MONO8);
      } catch (cv_bridge::Exception &e) {
        PRINT_ERROR(RED "cv_bridge exception: %s\n" RESET, e.what());
        continue;
      }
      has_left = true;
      preprocess_image(cv_ptr->image, img0, image_scale, max_width);
      time0 = c0->header.stamp.toSec();
    }

    // Handle RIGHT camera (raw image)
    sensor_msgs::Image::ConstPtr s1 = m.instantiate<sensor_msgs::Image>();
    if (s1 != nullptr && msg_topic == topic_camera1) {
      // Get the image
      cv_bridge::CvImageConstPtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvShare(s1, sensor_msgs::image_encodings::MONO8);
      } catch (cv_bridge::Exception &e) {
        PRINT_ERROR(RED "cv_bridge exception: %s\n" RESET, e.what());
        continue;
      }
      // Save to our temp variable
      has_right = true;
      preprocess_image(cv_ptr->image, img1, image_scale, max_width);
      time1 = cv_ptr->header.stamp.toSec();
    }
    // Handle RIGHT camera (compressed image)
    sensor_msgs::CompressedImage::ConstPtr c1 = m.instantiate<sensor_msgs::CompressedImage>();
    if (c1 != nullptr && msg_topic == topic_camera1_compressed) {
      cv_bridge::CvImagePtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvCopy(c1, sensor_msgs::image_encodings::MONO8);
      } catch (cv_bridge::Exception &e) {
        PRINT_ERROR(RED "cv_bridge exception: %s\n" RESET, e.what());
        continue;
      }
      has_right = true;
      preprocess_image(cv_ptr->image, img1, image_scale, max_width);
      time1 = c1->header.stamp.toSec();
    }

    // Process either stereo pair or mono frame
    if ((max_cameras == 2 && has_left && has_right) || (max_cameras == 1 && has_left)) {
      handle_stereo(time0, time1, img0, img1);
      processed_any = true;
      has_left = false;
      has_right = false;
    }
  }
#elif ROS_AVAILABLE == 2
  if (bag_storage_id == "auto") {
    if (ends_with_case_insensitive(path_to_bag, ".mcap")) {
      bag_storage_id = "mcap";
    } else if (ends_with_case_insensitive(path_to_bag, ".db3") || ends_with_case_insensitive(path_to_bag, ".sqlite3")) {
      bag_storage_id = "sqlite3";
    } else {
      bag_storage_id = "sqlite3";
    }
  }

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = path_to_bag;
  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  std::vector<std::string> storage_candidates;
  storage_candidates.push_back(bag_storage_id);
  if (bag_storage_id == "mcap") {
    storage_candidates.push_back("sqlite3");
  } else if (bag_storage_id == "sqlite3") {
    storage_candidates.push_back("mcap");
  }

  bool reader_opened = false;
  std::string open_errors;
  rosbag2_cpp::Reader reader;
  for (const auto &candidate : storage_candidates) {
    try {
      storage_options.storage_id = candidate;
      PRINT_INFO("trying rosbag2 storage backend: %s\n", candidate.c_str());
      reader.open(storage_options, converter_options);
      reader_opened = true;
      PRINT_INFO("opened bag with rosbag2 storage backend: %s\n", candidate.c_str());
      break;
    } catch (const std::exception &e) {
      open_errors += candidate + ": " + e.what() + "\n";
    }
  }
  if (!reader_opened) {
    PRINT_ERROR(RED "failed to open bag '%s' with tested backends:\n%s\n" RESET, path_to_bag.c_str(), open_errors.c_str());
    PRINT_ERROR(RED "try running with --ros-args -p bag_storage_id:=mcap (or :=sqlite3)\n" RESET);
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  const int64_t bag_start_offset_ns = static_cast<int64_t>(bag_start * 1e9);
  const int64_t bag_duration_ns = static_cast<int64_t>(bag_durr * 1e9);
  int64_t bag_window_start_ns = 0;
  int64_t bag_window_end_ns = std::numeric_limits<int64_t>::max();
  bool window_initialized = false;

  rclcpp::Serialization<sensor_msgs::msg::Image> serializer_image;
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serializer_compressed;

  while (reader.has_next()) {

    if (!rclcpp::ok()) {
      break;
    }

    auto msg = reader.read_next();
    const std::string topic = normalize_topic_name(msg->topic_name);
    const int64_t msg_time_ns =
        (msg->send_timestamp > 0) ? static_cast<int64_t>(msg->send_timestamp) : static_cast<int64_t>(msg->recv_timestamp);

    if (!window_initialized) {
      bag_window_start_ns = msg_time_ns + bag_start_offset_ns;
      if (bag_durr >= 0.0) {
        bag_window_end_ns = bag_window_start_ns + bag_duration_ns;
      }
      PRINT_DEBUG("time start = %.6f\n", static_cast<double>(bag_window_start_ns) * 1e-9);
      if (bag_durr >= 0.0) {
        PRINT_DEBUG("time end   = %.6f\n", static_cast<double>(bag_window_end_ns) * 1e-9);
      } else {
        PRINT_DEBUG("time end   = end_of_bag\n");
      }
      time0 = static_cast<double>(bag_window_start_ns) * 1e-9;
      time1 = time0;
      window_initialized = true;
    }

    if (msg_time_ns < bag_window_start_ns) {
      continue;
    }
    if (bag_durr >= 0.0 && msg_time_ns > bag_window_end_ns) {
      break;
    }

    const bool is_left_raw = (topic == topic_camera0);
    const bool is_left_compressed = (topic == topic_camera0_compressed);
    const bool is_right_raw = (topic == topic_camera1);
    const bool is_right_compressed = (topic == topic_camera1_compressed);
    if (!is_left_raw && !is_left_compressed && !is_right_raw && !is_right_compressed) {
      continue;
    }

    cv_bridge::CvImagePtr cv_ptr;
    double stamp_sec = 0.0;
    rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
    if (is_left_raw || is_right_raw) {
      sensor_msgs::msg::Image image_msg;
      serializer_image.deserialize_message(&serialized_msg, &image_msg);
      auto image_ptr = std::make_shared<sensor_msgs::msg::Image>(image_msg);
      try {
        cv_ptr = cv_bridge::toCvCopy(image_ptr, sensor_msgs::image_encodings::MONO8);
      } catch (cv_bridge::Exception &e) {
        PRINT_ERROR(RED "cv_bridge exception: %s\n" RESET, e.what());
        continue;
      }
      stamp_sec = static_cast<double>(image_ptr->header.stamp.sec) + static_cast<double>(image_ptr->header.stamp.nanosec) * 1e-9;
    } else {
      sensor_msgs::msg::CompressedImage image_msg;
      serializer_compressed.deserialize_message(&serialized_msg, &image_msg);
      auto image_ptr = std::make_shared<sensor_msgs::msg::CompressedImage>(image_msg);
      try {
        cv_ptr = cv_bridge::toCvCopy(image_ptr, sensor_msgs::image_encodings::MONO8);
      } catch (cv_bridge::Exception &e) {
        PRINT_ERROR(RED "cv_bridge exception: %s\n" RESET, e.what());
        continue;
      }
      stamp_sec = static_cast<double>(image_ptr->header.stamp.sec) + static_cast<double>(image_ptr->header.stamp.nanosec) * 1e-9;
    }
    if (is_left_raw || is_left_compressed) {
      has_left = true;
      preprocess_image(cv_ptr->image, img0, image_scale, max_width);
      time0 = stamp_sec;
    } else if (is_right_raw || is_right_compressed) {
      has_right = true;
      preprocess_image(cv_ptr->image, img1, image_scale, max_width);
      time1 = stamp_sec;
    }

    if ((max_cameras == 2 && has_left && has_right) || (max_cameras == 1 && has_left)) {
      handle_stereo(time0, time1, img0, img1);
      processed_any = true;
      has_left = false;
      has_right = false;
    }
  }

  if (!processed_any) {
    if (max_cameras == 1) {
      PRINT_ERROR(RED "No images found in bag window for topic:\n\t%s\n\t(or compressed: %s)\n" RESET, topic_camera0.c_str(),
                  topic_camera0_compressed.c_str());
    } else {
      PRINT_ERROR(RED "No synchronized image pairs found in bag window for topics:\n\t%s\n\t%s\n\t(or compressed: %s, %s)\n" RESET,
                  topic_camera0.c_str(), topic_camera1.c_str(), topic_camera0_compressed.c_str(), topic_camera1_compressed.c_str());
    }
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
#endif

  // Done!
  if (tracking_log_stream.is_open()) {
    tracking_log_stream.flush();
    tracking_log_stream.close();
  }
  return EXIT_SUCCESS;
}

/**
 * This function will process the new stereo pair with the extractor!
 */
void handle_stereo(double time0, double time1, cv::Mat img0, cv::Mat img1) {

  proc_width = img0.cols;
  proc_height = img0.rows;

  // Animate our dynamic mask moving
  // Very simple ball bounding around the screen example
  cv::Mat mask = cv::Mat::zeros(cv::Size(img0.cols, img0.rows), CV_8UC1);
  if (enable_dynamic_mask) {
    static cv::Point2f ball_center;
    static cv::Point2f ball_velocity;
    if (ball_velocity.x == 0 || ball_velocity.y == 0) {
      ball_center.x = (float)img0.cols / 2.0f;
      ball_center.y = (float)img0.rows / 2.0f;
      ball_velocity.x = 2.5;
      ball_velocity.y = 2.5;
    }
    ball_center += ball_velocity;
    if (ball_center.x < 0 || (int)ball_center.x > img0.cols)
      ball_velocity.x *= -1;
    if (ball_center.y < 0 || (int)ball_center.y > img0.rows)
      ball_velocity.y *= -1;
    cv::circle(mask, ball_center, 100, cv::Scalar(255), cv::FILLED);
  }

  // Process this new image
  ov_core::CameraData message;
  message.timestamp = time0;
  message.sensor_ids.push_back(0);
  message.images.push_back(img0);
  message.masks.push_back(mask);
  if (max_cameras == 2) {
    message.sensor_ids.push_back(1);
    message.images.push_back(img1);
    message.masks.push_back(mask);
  }
  extractor->feed_new_camera(message);

  // Display the resulting tracks
  cv::Mat img_active, img_history;
  extractor->display_active(img_active, 255, 0, 0, 0, 0, 255);
  extractor->display_history(img_history, 255, 255, 0, 255, 255, 255);

  // Show our image!
  cv::imshow("Active Tracks", img_active);
  cv::imshow("Track History", img_history);
  cv::waitKey(1);

  // Get lost tracks
  std::shared_ptr<FeatureDatabase> database = extractor->get_feature_database();
  std::vector<std::shared_ptr<Feature>> feats_lost = database->features_not_containing_newer(time0);
  const int lost_this_frame = (int)feats_lost.size();
  int lost_total_meas_this_frame = 0;
  num_lostfeats += lost_this_frame;

  // Mark theses feature pointers as deleted
  for (size_t i = 0; i < feats_lost.size(); i++) {
    // Total number of measurements
    int total_meas = 0;
    for (auto const &pair : feats_lost[i]->timestamps) {
      total_meas += (int)pair.second.size();
    }
    lost_total_meas_this_frame += total_meas;
    // Update stats
    featslengths += total_meas;
    feats_lost[i]->to_delete = true;
  }

  // Push back the current time, as a clone time
  clonetimes.push_back(time0);

  // Marginalized features if we have reached 5 frame tracks
  if ((int)clonetimes.size() >= clone_states) {
    // Remove features that have reached their max track length
    double margtime = clonetimes.at(0);
    clonetimes.pop_front();
    std::vector<std::shared_ptr<Feature>> feats_marg = database->features_containing(margtime);
    num_margfeats += feats_marg.size();
    const int marg_this_frame = (int)feats_marg.size();
    // Delete theses feature pointers
    for (size_t i = 0; i < feats_marg.size(); i++) {
      feats_marg[i]->to_delete = true;
    }
    tracking_log_sum_marg_feats += marg_this_frame;
  }

  // Tell the feature database to delete old features
  database->cleanup();

  // Active tracked features in latest frame observations.
  size_t track_feat = 0;
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> last_obs = extractor->get_last_obs();
  for (const auto &obs : last_obs) {
    track_feat += obs.second.size();
  }
  const size_t numtracks = database->size();

  if (tracking_log_enable) {
    if (tracking_log_frame_count == 0) {
      tracking_log_time_start = wall_clock_seconds();
    }
    tracking_log_frame_count++;
    tracking_log_sum_track_feat += static_cast<double>(track_feat);
    tracking_log_sum_numtracks += static_cast<double>(numtracks);
    tracking_log_sum_lost_feats += static_cast<double>(lost_this_frame);
    tracking_log_sum_lost_track_measurements += static_cast<double>(lost_total_meas_this_frame);
    if (tracking_log_frame_count >= tracking_log_avg_frames) {
      const double now = wall_clock_seconds();
      const double dt = std::max(1e-9, now - tracking_log_time_start);
      const double fps_window = static_cast<double>(tracking_log_frame_count) / dt;
      const double avg_track_feat = tracking_log_sum_track_feat / tracking_log_frame_count;
      const double avg_numtracks = tracking_log_sum_numtracks / tracking_log_frame_count;
      const double avg_lost = tracking_log_sum_lost_feats / tracking_log_frame_count;
      const double avg_marg = tracking_log_sum_marg_feats / tracking_log_frame_count;
      const double avg_track_length_lost_feat =
          (tracking_log_sum_lost_feats > 0.0) ? (tracking_log_sum_lost_track_measurements / tracking_log_sum_lost_feats) : 0.0;
      std::ostringstream ss;
      ss << "avg,"
         << tracking_log_frame_count << ","
         << fps_window << ","
         << avg_track_feat << ","
         << avg_numtracks << ","
         << avg_lost << ","
         << avg_marg << ","
         << avg_track_length_lost_feat;
      tracking_log_write(ss.str());
      tracking_log_frame_count = 0;
      tracking_log_sum_track_feat = 0.0;
      tracking_log_sum_numtracks = 0.0;
      tracking_log_sum_lost_feats = 0.0;
      tracking_log_sum_marg_feats = 0.0;
      tracking_log_sum_lost_track_measurements = 0.0;
    }
  }

  // Debug print out what our current processing speed it
  // We want the FPS to be as high as possible
  const double time_curr = wall_clock_seconds();
  if (frames > 60) {
    // Calculate the FPS
    double fps = (double)frames / (time_curr - time_start);
    double lpf = (double)num_lostfeats / frames;
    double fpf = (double)featslengths / num_lostfeats;
    double mpf = (double)num_margfeats / frames;
    // DEBUG PRINT OUT
    PRINT_DEBUG("res = %dx%d | fps = %.2f | lost_feats/frame = %.2f | track_length/lost_feat = %.2f | marg_tracks/frame = %.2f\n", proc_width,
                proc_height, fps, lpf, fpf, mpf);
    // Reset variables
    frames = 0;
    time_start = time_curr;
    num_lostfeats = 0;
    num_margfeats = 0;
    featslengths = 0;
  }
  frames++;
}
