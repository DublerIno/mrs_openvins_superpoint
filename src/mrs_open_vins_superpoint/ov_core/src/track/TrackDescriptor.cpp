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

#include "TrackDescriptor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <sys/wait.h>
#include <signal.h>

#include <opencv2/features2d.hpp>

#include "Grider_FAST.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"

using namespace ov_core;

namespace {

bool write_all(int fd, const uint8_t *data, size_t size) {
  size_t written = 0;
  while (written < size) {
    ssize_t ret = ::write(fd, data + written, size - written);
    if (ret <= 0) {
      return false;
    }
    written += static_cast<size_t>(ret);
  }
  return true;
}

bool read_all(int fd, uint8_t *data, size_t size) {
  size_t read_total = 0;
  while (read_total < size) {
    ssize_t ret = ::read(fd, data + read_total, size - read_total);
    if (ret <= 0) {
      return false;
    }
    read_total += static_cast<size_t>(ret);
  }
  return true;
}

bool env_var_true(const char *v) {
  if (v == nullptr) {
    return false;
  }
  return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0 ||
         std::strcmp(v, "yes") == 0 || std::strcmp(v, "YES") == 0 || std::strcmp(v, "on") == 0 || std::strcmp(v, "ON") == 0;
}

} // namespace

TrackDescriptor::~TrackDescriptor() {
  stop_superpoint_worker();
}

bool TrackDescriptor::start_superpoint_worker() {
  if (sp_worker_started) {
    return true;
  }
  if (sp_worker_failed || sp_weights_path.empty()) {
    return false;
  }

  std::string script_path = __FILE__;
  const size_t last_slash = script_path.find_last_of('/');
  if (last_slash == std::string::npos) {
    sp_worker_failed = true;
    return false;
  }
  script_path = script_path.substr(0, last_slash + 1) + "superpoint_pytorch_worker.py";
  std::ifstream script_file(script_path);
  if (!script_file.good()) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker script missing: %s\n" RESET, script_path.c_str());
    sp_worker_failed = true;
    return false;
  }

  int to_worker[2] = {-1, -1};
  int from_worker[2] = {-1, -1};
  if (pipe(to_worker) != 0 || pipe(from_worker) != 0) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: failed to create pipes for python worker\n" RESET);
    if (to_worker[0] >= 0) close(to_worker[0]);
    if (to_worker[1] >= 0) close(to_worker[1]);
    if (from_worker[0] >= 0) close(from_worker[0]);
    if (from_worker[1] >= 0) close(from_worker[1]);
    sp_worker_failed = true;
    return false;
  }

  const char *python_env = std::getenv("OV_SUPERPOINT_PYTHON");
  const std::string python_bin = (python_env != nullptr && python_env[0] != '\0') ? std::string(python_env) : std::string("python3");
  const std::string conf_str = std::to_string(sp_threshold);
  const std::string nms_str = std::to_string(sp_do_nms ? 4 : 0);
  const std::string cuda_str = std::to_string(sp_use_cuda ? 1 : 0);
  const std::string nfeatures_str = std::to_string(sp_nfeatures);
  const char *sg_weights_env = std::getenv("OV_SUPERGLUE_WEIGHTS");
  const char *sg_match_env = std::getenv("OV_SUPERGLUE_MATCH_THRESHOLD");
  const char *sg_sinkhorn_env = std::getenv("OV_SUPERGLUE_SINKHORN_ITERATIONS");
  const std::string sg_weights = (sg_weights_env != nullptr && sg_weights_env[0] != '\0') ? std::string(sg_weights_env) : std::string("outdoor");
  const std::string sg_match = (sg_match_env != nullptr && sg_match_env[0] != '\0') ? std::string(sg_match_env) : std::string("0.2");
  const std::string sg_sinkhorn = (sg_sinkhorn_env != nullptr && sg_sinkhorn_env[0] != '\0') ? std::string(sg_sinkhorn_env) : std::string("20");

  pid_t pid = fork();
  if (pid < 0) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: fork failed for python worker\n" RESET);
    close(to_worker[0]);
    close(to_worker[1]);
    close(from_worker[0]);
    close(from_worker[1]);
    sp_worker_failed = true;
    return false;
  }

  if (pid == 0) {
    dup2(to_worker[0], STDIN_FILENO);
    dup2(from_worker[1], STDOUT_FILENO);
    close(to_worker[0]);
    close(to_worker[1]);
    close(from_worker[0]);
    close(from_worker[1]);

    execlp(python_bin.c_str(), python_bin.c_str(), script_path.c_str(),
           "--weights", sp_weights_path.c_str(),
           "--conf_thresh", conf_str.c_str(),
           "--nms_dist", nms_str.c_str(),
           "--cuda", cuda_str.c_str(),
           "--num_features", nfeatures_str.c_str(),
           "--superglue", sg_weights.c_str(),
           "--sinkhorn_iterations", sg_sinkhorn.c_str(),
           "--match_threshold", sg_match.c_str(),
           (char *)nullptr);
    _exit(127);
  }

  close(to_worker[0]);
  close(from_worker[1]);
  sp_worker_stdin_fd = to_worker[1];
  sp_worker_stdout_fd = from_worker[0];
  sp_worker_pid = pid;
  sp_worker_started = true;
  PRINT_INFO("[TRACK-DESC]: started persistent python SuperPoint worker (pid=%d)\n", (int)sp_worker_pid);
  return true;
}

void TrackDescriptor::stop_superpoint_worker_nolock() {
  if (sp_worker_stdin_fd >= 0) {
    close(sp_worker_stdin_fd);
    sp_worker_stdin_fd = -1;
  }
  if (sp_worker_stdout_fd >= 0) {
    close(sp_worker_stdout_fd);
    sp_worker_stdout_fd = -1;
  }
  if (sp_worker_pid > 0) {
    kill(sp_worker_pid, SIGTERM);
    waitpid(sp_worker_pid, nullptr, 0);
    sp_worker_pid = -1;
  }
  sp_worker_started = false;
}

void TrackDescriptor::stop_superpoint_worker() {
  std::lock_guard<std::mutex> lock(sp_worker_mtx);
  stop_superpoint_worker_nolock();
}

bool TrackDescriptor::run_superpoint_worker(const cv::Mat &img, std::vector<cv::KeyPoint> &pts_out, cv::Mat &desc_out,
                                            std::vector<float> &scores_out) {
  std::lock_guard<std::mutex> lock(sp_worker_mtx);

  if (img.empty()) {
    return false;
  }
  if (!start_superpoint_worker()) {
    return false;
  }

  cv::Mat gray = (img.type() == CV_8UC1) ? img : cv::Mat();
  if (gray.empty()) {
    img.convertTo(gray, CV_8U);
  }
  if (!gray.isContinuous()) {
    gray = gray.clone();
  }

  const uint32_t width = static_cast<uint32_t>(gray.cols);
  const uint32_t height = static_cast<uint32_t>(gray.rows);
  const size_t image_bytes = static_cast<size_t>(width) * static_cast<size_t>(height);
  std::array<uint32_t, 3> req_header = {0u, width, height};

  if (!write_all(sp_worker_stdin_fd, reinterpret_cast<const uint8_t *>(req_header.data()), sizeof(req_header)) ||
      !write_all(sp_worker_stdin_fd, gray.data, image_bytes)) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker write failed\n" RESET);
    stop_superpoint_worker_nolock();
    sp_worker_failed = true;
    return false;
  }

  std::array<uint32_t, 3> resp_header = {0, 0, 0}; // status, num_points, desc_dim
  if (!read_all(sp_worker_stdout_fd, reinterpret_cast<uint8_t *>(resp_header.data()), sizeof(resp_header))) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker read header failed\n" RESET);
    stop_superpoint_worker_nolock();
    sp_worker_failed = true;
    return false;
  }

  const uint32_t status = resp_header[0];
  const uint32_t num_points = resp_header[1];
  const uint32_t desc_dim = resp_header[2];
  if (status != 0) {
    return false;
  }

  std::vector<float> points_xy(static_cast<size_t>(num_points) * 2, 0.0f);
  if (!points_xy.empty() &&
      !read_all(sp_worker_stdout_fd, reinterpret_cast<uint8_t *>(points_xy.data()), points_xy.size() * sizeof(float))) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker read points failed\n" RESET);
    stop_superpoint_worker_nolock();
    sp_worker_failed = true;
    return false;
  }

  pts_out.clear();
  desc_out.release();
  scores_out.clear();
  pts_out.reserve(num_points);
  scores_out.reserve(num_points);
  for (uint32_t i = 0; i < num_points; i++) {
    pts_out.emplace_back(cv::Point2f(points_xy[2 * i], points_xy[2 * i + 1]), 1.0f);
  }

  if (num_points == 0) {
    return true;
  }

  if (desc_dim > 0) {
    desc_out = cv::Mat((int)num_points, (int)desc_dim, CV_32F);
    const size_t desc_bytes = static_cast<size_t>(num_points) * static_cast<size_t>(desc_dim) * sizeof(float);
    if (!read_all(sp_worker_stdout_fd, reinterpret_cast<uint8_t *>(desc_out.data), desc_bytes)) {
      PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker read descriptors failed\n" RESET);
      stop_superpoint_worker_nolock();
      sp_worker_failed = true;
      return false;
    }
  }

  scores_out.assign(static_cast<size_t>(num_points), 1.0f);
  if (num_points > 0 &&
      !read_all(sp_worker_stdout_fd, reinterpret_cast<uint8_t *>(scores_out.data()), static_cast<size_t>(num_points) * sizeof(float))) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker read scores failed\n" RESET);
    stop_superpoint_worker_nolock();
    sp_worker_failed = true;
    return false;
  }

  return true;
}

bool TrackDescriptor::run_superglue_worker(const cv::Mat &img0, const cv::Mat &img1, const std::vector<cv::KeyPoint> &pts0,
                                           const std::vector<cv::KeyPoint> &pts1, const cv::Mat &desc0, const cv::Mat &desc1,
                                           const std::vector<float> &scores0, const std::vector<float> &scores1,
                                           std::vector<int> &matches0, std::vector<float> &matching_scores0) {
  std::lock_guard<std::mutex> lock(sp_worker_mtx);

  if (img0.empty() || img1.empty()) {
    return false;
  }
  if (!start_superpoint_worker()) {
    return false;
  }
  if (desc0.type() != CV_32F || desc1.type() != CV_32F) {
    return false;
  }
  if (desc0.rows != (int)pts0.size() || desc1.rows != (int)pts1.size()) {
    return false;
  }
  if (desc0.cols != desc1.cols) {
    return false;
  }

  cv::Mat gray0 = (img0.type() == CV_8UC1) ? img0 : cv::Mat();
  if (gray0.empty()) {
    img0.convertTo(gray0, CV_8U);
  }
  cv::Mat gray1 = (img1.type() == CV_8UC1) ? img1 : cv::Mat();
  if (gray1.empty()) {
    img1.convertTo(gray1, CV_8U);
  }
  if (!gray0.isContinuous()) {
    gray0 = gray0.clone();
  }
  if (!gray1.isContinuous()) {
    gray1 = gray1.clone();
  }

  const uint32_t w0 = static_cast<uint32_t>(gray0.cols);
  const uint32_t h0 = static_cast<uint32_t>(gray0.rows);
  const uint32_t w1 = static_cast<uint32_t>(gray1.cols);
  const uint32_t h1 = static_cast<uint32_t>(gray1.rows);
  const uint32_t n0 = static_cast<uint32_t>(pts0.size());
  const uint32_t n1 = static_cast<uint32_t>(pts1.size());
  const uint32_t d = static_cast<uint32_t>(desc0.cols);

  std::vector<float> pts0_xy(static_cast<size_t>(n0) * 2, 0.0f);
  std::vector<float> pts1_xy(static_cast<size_t>(n1) * 2, 0.0f);
  for (uint32_t i = 0; i < n0; i++) {
    pts0_xy[2 * i] = pts0[i].pt.x;
    pts0_xy[2 * i + 1] = pts0[i].pt.y;
  }
  for (uint32_t i = 0; i < n1; i++) {
    pts1_xy[2 * i] = pts1[i].pt.x;
    pts1_xy[2 * i + 1] = pts1[i].pt.y;
  }

  std::vector<float> s0 = scores0;
  std::vector<float> s1 = scores1;
  if (s0.size() != (size_t)n0) {
    s0.assign((size_t)n0, 1.0f);
  }
  if (s1.size() != (size_t)n1) {
    s1.assign((size_t)n1, 1.0f);
  }

  std::array<uint32_t, 8> req_header = {1u, w0, h0, w1, h1, n0, n1, d};
  if (!write_all(sp_worker_stdin_fd, reinterpret_cast<const uint8_t *>(req_header.data()), sizeof(req_header)) ||
      !write_all(sp_worker_stdin_fd, gray0.data, static_cast<size_t>(w0) * static_cast<size_t>(h0)) ||
      !write_all(sp_worker_stdin_fd, gray1.data, static_cast<size_t>(w1) * static_cast<size_t>(h1))) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker write for SuperGlue failed\n" RESET);
    stop_superpoint_worker_nolock();
    sp_worker_failed = true;
    return false;
  }

  if ((!pts0_xy.empty() &&
       !write_all(sp_worker_stdin_fd, reinterpret_cast<const uint8_t *>(pts0_xy.data()), pts0_xy.size() * sizeof(float))) ||
      (!pts1_xy.empty() &&
       !write_all(sp_worker_stdin_fd, reinterpret_cast<const uint8_t *>(pts1_xy.data()), pts1_xy.size() * sizeof(float))) ||
      (!s0.empty() && !write_all(sp_worker_stdin_fd, reinterpret_cast<const uint8_t *>(s0.data()), s0.size() * sizeof(float))) ||
      (!s1.empty() && !write_all(sp_worker_stdin_fd, reinterpret_cast<const uint8_t *>(s1.data()), s1.size() * sizeof(float)))) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker write keypoints/scores for SuperGlue failed\n" RESET);
    stop_superpoint_worker_nolock();
    sp_worker_failed = true;
    return false;
  }

  const size_t desc0_bytes = static_cast<size_t>(desc0.rows) * static_cast<size_t>(desc0.cols) * sizeof(float);
  const size_t desc1_bytes = static_cast<size_t>(desc1.rows) * static_cast<size_t>(desc1.cols) * sizeof(float);
  if ((desc0_bytes > 0 && !write_all(sp_worker_stdin_fd, reinterpret_cast<const uint8_t *>(desc0.data), desc0_bytes)) ||
      (desc1_bytes > 0 && !write_all(sp_worker_stdin_fd, reinterpret_cast<const uint8_t *>(desc1.data), desc1_bytes))) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker write descriptors for SuperGlue failed\n" RESET);
    stop_superpoint_worker_nolock();
    sp_worker_failed = true;
    return false;
  }

  std::array<uint32_t, 2> resp_header = {0, 0}; // status, num_points
  if (!read_all(sp_worker_stdout_fd, reinterpret_cast<uint8_t *>(resp_header.data()), sizeof(resp_header))) {
    PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker read SuperGlue header failed\n" RESET);
    stop_superpoint_worker_nolock();
    sp_worker_failed = true;
    return false;
  }
  if (resp_header[0] != 0) {
    return false;
  }
  const uint32_t out_n0 = resp_header[1];
  if (out_n0 != n0) {
    return false;
  }

  std::vector<int32_t> matches0_i32((size_t)out_n0, -1);
  matching_scores0.assign((size_t)out_n0, 0.0f);
  if (out_n0 > 0) {
    if (!read_all(sp_worker_stdout_fd, reinterpret_cast<uint8_t *>(matches0_i32.data()), static_cast<size_t>(out_n0) * sizeof(int32_t)) ||
        !read_all(sp_worker_stdout_fd, reinterpret_cast<uint8_t *>(matching_scores0.data()),
                  static_cast<size_t>(out_n0) * sizeof(float))) {
      PRINT_WARNING(YELLOW "[TRACK-DESC]: python worker read SuperGlue matches failed\n" RESET);
      stop_superpoint_worker_nolock();
      sp_worker_failed = true;
      return false;
    }
  }
  matches0.assign(matches0_i32.begin(), matches0_i32.end());

  return true;
}

//NEW Function - chooses camera based on image message
void TrackDescriptor::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // Either call our stereo or monocular version
  // If we are doing binocular tracking, then we should parallize our tracking
  size_t num_images = message.images.size();
  if (num_images == 1) {
    feed_monocular(message, 0);
  } else if (num_images == 2 && use_stereo) {
    feed_stereo(message, 0, 1);
  } else if (!use_stereo) {
    parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
                    for (int i = range.start; i < range.end; i++) {
                      feed_monocular(message, i);
                    }
                  }));
  } else {
    PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
    std::exit(EXIT_FAILURE);
  }
}

void TrackDescriptor::feed_monocular(const CameraData &message, size_t msg_id) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Histogram equalize
  cv::Mat img, mask;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id), img);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id), img);
  } else {
    img = message.images.at(msg_id);
  }
  mask = message.masks.at(msg_id);

  // If we are the first frame (or have lost tracking), initialize our descriptors
  if (pts_last.find(cam_id) == pts_last.end() || pts_last[cam_id].empty()) {
    std::vector<cv::KeyPoint> good_left;
    std::vector<cv::KeyPoint> raw_left;
    std::vector<float> scores_left;
    std::vector<size_t> good_ids_left;
    cv::Mat good_desc_left;
    perform_detection_monocular(img, mask, good_left, good_desc_left, good_ids_left, &raw_left, &scores_left);
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    pts_last_raw[cam_id] = raw_left; //for raw visualtization
    ids_last[cam_id] = good_ids_left;
    desc_last[cam_id] = good_desc_left;
    scores_last[cam_id] = scores_left;
    PRINT_DEBUG("Initialized first frame for cam_id %zu with %zu features\n", cam_id, good_left.size());
    return;
  }

  // Our new keypoints and descriptor for the new image
  std::vector<cv::KeyPoint> pts_new;
  std::vector<cv::KeyPoint> pts_new_raw; //for raw visualtization
  std::vector<float> scores_new;
  cv::Mat desc_new;
  std::vector<size_t> ids_new;

  // First, extract new descriptors for this new image
  perform_detection_monocular(img, mask, pts_new, desc_new, ids_new, &pts_new_raw, &scores_new);

  //before perform_detection_monocular(img, pts_new, desc_new, ids_new);

  rT2 = boost::posix_time::microsec_clock::local_time();


  // Match
  // Our matches temporally left to left
  std::vector<cv::DMatch> matches_ll;

  // Lets match temporally
  robust_match(pts_last[cam_id], pts_new, desc_last[cam_id], desc_new, cam_id, cam_id, matches_ll, &img_last[cam_id], &img,
               &scores_last[cam_id], &scores_new);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;
  cv::Mat good_desc_left;

  // Count how many we have tracked from the last time
  int num_tracklast = 0;

  // Loop through all current left to right points
  // We want to see if any of theses have matches to the previous frame
  // If we have a match new->old then we want to use that ID instead of the new one
  for (size_t i = 0; i < pts_new.size(); i++) { //for all new features

    // Loop through all left matches, and find the old "train" id
    int idll = -1;
    for (size_t j = 0; j < matches_ll.size(); j++) {
      if (matches_ll[j].trainIdx == (int)i) {
        idll = matches_ll[j].queryIdx;
      }
    } 
    
    //if we found a good track from left to left
    // Then lets replace the current ID with the old ID if found
    // Else just append the current feature and its unique ID
    good_left.push_back(pts_new[i]);
    good_desc_left.push_back(desc_new.row((int)i));
    if (idll != -1) {
      good_ids_left.push_back(ids_last[cam_id][idll]);
      num_tracklast++;
    } else {
      good_ids_left.push_back(ids_new[i]);
    }
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Update our feature database, with theses new observations
  // main output of the class
  for (size_t i = 0; i < good_left.size(); i++) { //for all good features
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  // Debug info
   PRINT_DEBUG("LtoL = %d | good = %d | fromlast = %d\n",(int)matches_ll.size(),(int)good_left.size(),num_tracklast);

  // Move forward in time
  // Save current as last - lock only for this small amount of time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars); //calls mtx_last_vars.lock(); and mtx_last_vars.unlock() when we leave this scope
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    pts_last_raw[cam_id] = pts_new_raw;
    ids_last[cam_id] = good_ids_left;
    desc_last[cam_id] = good_desc_left;
    scores_last[cam_id] = scores_new;
  }
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Our timing information
  PRINT_ALL("[TIME-DESC]: %.4f seconds for detection\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for matching\n", (rT3 - rT2).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for merging\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-DESC]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void TrackDescriptor::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Histogram equalize images
  cv::Mat img_left, img_right, mask_left, mask_right;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id_left), img_left);
    cv::equalizeHist(message.images.at(msg_id_right), img_right);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id_left), img_left);
    clahe->apply(message.images.at(msg_id_right), img_right);
  } else {
    img_left = message.images.at(msg_id_left);
    img_right = message.images.at(msg_id_right);
  }
  mask_left = message.masks.at(msg_id_left);
  mask_right = message.masks.at(msg_id_right);

  // If we are the first frame (or have lost tracking), initialize our descriptors
  if (pts_last[cam_id_left].empty() || pts_last[cam_id_right].empty()) {
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;
    cv::Mat good_desc_left, good_desc_right;
    perform_detection_stereo(img_left, img_right, mask_left, mask_right, good_left, good_right, good_desc_left, good_desc_right,
                             cam_id_left, cam_id_right, good_ids_left, good_ids_right);
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    desc_last[cam_id_left] = good_desc_left;
    desc_last[cam_id_right] = good_desc_right;
    return;
  }

  // Our new keypoints and descriptor for the new image
  std::vector<cv::KeyPoint> pts_left_new, pts_right_new;
  cv::Mat desc_left_new, desc_right_new;
  std::vector<size_t> ids_left_new, ids_right_new;

  // First, extract new descriptors for this new image
  perform_detection_stereo(img_left, img_right, mask_left, mask_right, pts_left_new, pts_right_new, desc_left_new, desc_right_new,
                           cam_id_left, cam_id_right, ids_left_new, ids_right_new);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Our matches temporally
  std::vector<cv::DMatch> matches_ll, matches_rr;
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    robust_match(pts_last[is_left ? cam_id_left : cam_id_right], is_left ? pts_left_new : pts_right_new,
                                 desc_last[is_left ? cam_id_left : cam_id_right], is_left ? desc_left_new : desc_right_new,
                                 is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                                 is_left ? matches_ll : matches_rr);
                  }
                }));
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;
  cv::Mat good_desc_left, good_desc_right;

  // Points must be of equal size
  assert(pts_last[cam_id_left].size() == pts_last[cam_id_right].size());
  assert(pts_left_new.size() == pts_right_new.size());

  // Count how many we have tracked from the last time
  int num_tracklast = 0;

  // Loop through all current left to right points
  // We want to see if any of theses have matches to the previous frame
  // If we have a match new->old then we want to use that ID instead of the new one
  for (size_t i = 0; i < pts_left_new.size(); i++) {

    // Loop through all left matches, and find the old "train" id
    int idll = -1;
    for (size_t j = 0; j < matches_ll.size(); j++) {
      if (matches_ll[j].trainIdx == (int)i) {
        idll = matches_ll[j].queryIdx;
      }
    }

    // Loop through all left matches, and find the old "train" id
    int idrr = -1;
    for (size_t j = 0; j < matches_rr.size(); j++) {
      if (matches_rr[j].trainIdx == (int)i) {
        idrr = matches_rr[j].queryIdx;
      }
    }

    // If we found a good stereo track from left to left, and right to right
    // Then lets replace the current ID with the old ID
    // We also check that we are linked to the same past ID value
    if (idll != -1 && idrr != -1 && ids_last[cam_id_left][idll] == ids_last[cam_id_right][idrr]) {
      good_left.push_back(pts_left_new[i]);
      good_right.push_back(pts_right_new[i]);
      good_desc_left.push_back(desc_left_new.row((int)i));
      good_desc_right.push_back(desc_right_new.row((int)i));
      good_ids_left.push_back(ids_last[cam_id_left][idll]);
      good_ids_right.push_back(ids_last[cam_id_right][idrr]);
      num_tracklast++;
    } else {
      // Else just append the current feature and its unique ID
      good_left.push_back(pts_left_new[i]);
      good_right.push_back(pts_right_new[i]);
      good_desc_left.push_back(desc_left_new.row((int)i));
      good_desc_right.push_back(desc_right_new.row((int)i));
      good_ids_left.push_back(ids_left_new[i]);
      good_ids_right.push_back(ids_left_new[i]);
    }
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    // Assert that our IDs are the same
    assert(good_ids_left.at(i) == good_ids_right.at(i));
    // Try to undistort the point
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    // Append to the database
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
  }

  // Debug info
  // PRINT_DEBUG("LtoL = %d | RtoR = %d | LtoR = %d | good = %d | fromlast = %d\n", (int)matches_ll.size(),
  //       (int)matches_rr.size(),(int)ids_left_new.size(),(int)good_left.size(),num_tracklast);

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    desc_last[cam_id_left] = good_desc_left;
    desc_last[cam_id_right] = good_desc_right;
  }
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Our timing information
  PRINT_ALL("[TIME-DESC]: %.4f seconds for detection\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for matching\n", (rT3 - rT2).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for merging\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-DESC]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

//===================Superpoint implementation===================
void TrackDescriptor::perform_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                                  cv::Mat &desc0, std::vector<size_t> &ids0, std::vector<cv::KeyPoint> *pts0_raw,
                                                  std::vector<float> *scores0) {

  // Assert that we need features
  assert(pts0.empty());
  if (scores0 != nullptr) {
    scores0->clear();
  }
  //SUPERPOINT = GET both keypoints and descriptors
  
  // For all new points, extract their descriptors
  cv::Mat desc0_ext;
  std::vector<cv::KeyPoint> pts0_ext;
  std::vector<float> scores0_ext;
  const bool python_ok = run_superpoint_worker(img0, pts0_ext, desc0_ext, scores0_ext);
  if (!python_ok) {
    PRINT_ERROR(RED "[TRACK-DESC]: python SuperPoint worker failed. Fallback is disabled.\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  if (pts0_raw != nullptr)
    *pts0_raw = pts0_ext;

  // For all good matches, append to our returned vectors.
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features.
  // NOTE: this is due to the fact that we select update features based on feat id.
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with.
  const size_t usable_count = std::min(pts0_ext.size(), static_cast<size_t>(desc0_ext.rows));

  // Optional occupancy-grid enforcement (same idea as original ORB mono path).
  cv::Mat grid_2d;
  bool use_grid = sp_enforce_grid && min_px_dist > 0;
  if (use_grid) {
    const int grid_w = std::max(1, (int)((float)img0.cols / (float)min_px_dist));
    const int grid_h = std::max(1, (int)((float)img0.rows / (float)min_px_dist));
    grid_2d = cv::Mat::zeros(cv::Size(grid_w, grid_h), CV_8UC1);
  }

  for (size_t i = 0; i < usable_count; i++) {
    cv::KeyPoint kpt = pts0_ext.at(i);
    const int x = (int)kpt.pt.x;
    const int y = (int)kpt.pt.y;
    if (x < 0 || x >= img0.cols || y < 0 || y >= img0.rows) {
      continue;
    }

    // Mask logic used here instead of inside the python extractor.
    if (!mask0.empty() && mask0.at<uint8_t>(y, x) > 127) {
      continue;
    }

    if (use_grid) {
      const int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      const int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= grid_2d.cols || y_grid < 0 || y_grid >= grid_2d.rows) {
        continue;
      }
      if (grid_2d.at<uint8_t>(y_grid, x_grid) > 127) {
        continue;
      }
      grid_2d.at<uint8_t>(y_grid, x_grid) = 255;
    }

    pts0.push_back(kpt);
    desc0.push_back(desc0_ext.row((int)i));
    if (scores0 != nullptr) {
      const float score = (i < scores0_ext.size()) ? scores0_ext[i] : 1.0f;
      scores0->push_back(score);
    }
    size_t temp = ++currid;
    ids0.push_back(temp);
  }
}

//dont care 
void TrackDescriptor::perform_detection_stereo(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &mask0, const cv::Mat &mask1,
                                               std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, cv::Mat &desc0,
                                               cv::Mat &desc1, size_t cam_id0, size_t cam_id1, std::vector<size_t> &ids0,
                                               std::vector<size_t> &ids1) {

  // Assert that we need features
  assert(pts0.empty());
  assert(pts1.empty());

  // Extract our features (use FAST with griding), and their descriptors
  std::vector<cv::KeyPoint> pts0_ext, pts1_ext;
  cv::Mat desc0_ext, desc1_ext;
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    Grider_FAST::perform_griding(is_left ? img0 : img1, is_left ? mask0 : mask1, is_left ? pts0_ext : pts1_ext,
                                                 num_features, grid_x, grid_y, threshold, true);
                    (is_left ? orb0 : orb1)->compute(is_left ? img0 : img1, is_left ? pts0_ext : pts1_ext, is_left ? desc0_ext : desc1_ext);
                  }
                }));

  // Do matching from the left to the right image
  std::vector<cv::DMatch> matches;
  robust_match(pts0_ext, pts1_ext, desc0_ext, desc1_ext, cam_id0, cam_id1, matches);

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size0((int)((float)img0.cols / (float)min_px_dist), (int)((float)img0.rows / (float)min_px_dist));
  cv::Mat grid_2d_0 = cv::Mat::zeros(size0, CV_8UC1);
  cv::Size size1((int)((float)img1.cols / (float)min_px_dist), (int)((float)img1.rows / (float)min_px_dist));
  cv::Mat grid_2d_1 = cv::Mat::zeros(size1, CV_8UC1);

  // For all good matches, lets append to our returned vectors
  for (size_t i = 0; i < matches.size(); i++) {

    // Get our ids
    int index_pt0 = matches.at(i).queryIdx;
    int index_pt1 = matches.at(i).trainIdx;

    // Get current left/right keypoint, check that it is in bounds
    cv::KeyPoint kpt0 = pts0_ext.at(index_pt0);
    cv::KeyPoint kpt1 = pts1_ext.at(index_pt1);
    int x0 = (int)kpt0.pt.x;
    int y0 = (int)kpt0.pt.y;
    int x0_grid = (int)(kpt0.pt.x / (float)min_px_dist);
    int y0_grid = (int)(kpt0.pt.y / (float)min_px_dist);
    if (x0_grid < 0 || x0_grid >= size0.width || y0_grid < 0 || y0_grid >= size0.height || x0 < 0 || x0 >= img0.cols || y0 < 0 ||
        y0 >= img0.rows) {
      continue;
    }
    int x1 = (int)kpt1.pt.x;
    int y1 = (int)kpt1.pt.y;
    int x1_grid = (int)(kpt1.pt.x / (float)min_px_dist);
    int y1_grid = (int)(kpt1.pt.y / (float)min_px_dist);
    if (x1_grid < 0 || x1_grid >= size1.width || y1_grid < 0 || y1_grid >= size1.height || x1 < 0 || x1 >= img0.cols || y1 < 0 ||
        y1 >= img0.rows) {
      continue;
    }

    // Check if this keypoint is near another point
    if (grid_2d_0.at<uint8_t>(y0_grid, x0_grid) > 127 || grid_2d_1.at<uint8_t>(y1_grid, x1_grid) > 127)
      continue;

    // Append our keypoints and descriptors
    pts0.push_back(pts0_ext[index_pt0]);
    pts1.push_back(pts1_ext[index_pt1]);
    desc0.push_back(desc0_ext.row(index_pt0));
    desc1.push_back(desc1_ext.row(index_pt1));

    // Set our IDs to be unique IDs here, will later replace with corrected ones, after temporal matching
    size_t temp = ++currid;
    ids0.push_back(temp);
    ids1.push_back(temp);
  }
}

// will it work with superpoimnt descriptors?
void TrackDescriptor::robust_match(const std::vector<cv::KeyPoint> &pts0, const std::vector<cv::KeyPoint> &pts1, const cv::Mat &desc0,
                                   const cv::Mat &desc1, size_t id0, size_t id1, std::vector<cv::DMatch> &matches,
                                   const cv::Mat *img0, const cv::Mat *img1, const std::vector<float> *scores0,
                                   const std::vector<float> *scores1) {

  if (pts0.empty() || pts1.empty() || desc0.empty() || desc1.empty()) {
    PRINT_ALL("robust match error\n");
    return;
  }

  if (desc0.rows != (int)pts0.size() || desc1.rows != (int)pts1.size()) {
    PRINT_ALL("robust match error\n");
    return;
  }

  if (img0 != nullptr && img1 != nullptr && !img0->empty() && !img1->empty() && desc0.type() == CV_32F && desc1.type() == CV_32F) {
    const char *sg_enable_env = std::getenv("OV_SUPERGLUE_ENABLE");
    const bool superglue_enabled = (sg_enable_env == nullptr) ? true : env_var_true(sg_enable_env);
    if (!superglue_enabled) {
      // SuperGlue disabled explicitly by environment variable, use legacy KNN matcher path.
    } else {
    std::vector<int> sg_matches0;
    std::vector<float> sg_scores0;
    std::vector<float> default_scores0(pts0.size(), 1.0f);
    std::vector<float> default_scores1(pts1.size(), 1.0f);
    const std::vector<float> &in_scores0 = (scores0 != nullptr) ? *scores0 : default_scores0;
    const std::vector<float> &in_scores1 = (scores1 != nullptr) ? *scores1 : default_scores1;

    if (run_superglue_worker(*img0, *img1, pts0, pts1, desc0, desc1, in_scores0, in_scores1, sg_matches0, sg_scores0)) {
      std::vector<cv::DMatch> matches_good;
      matches_good.reserve(sg_matches0.size());
      for (size_t i = 0; i < sg_matches0.size(); i++) {
        const int midx = sg_matches0[i];
        if (midx < 0 || midx >= (int)pts1.size()) {
          continue;
        }
        const float dist = (i < sg_scores0.size()) ? (1.0f - sg_scores0[i]) : 0.0f;
        matches_good.emplace_back(cv::DMatch((int)i, midx, dist));
      }

      std::vector<cv::Point2f> pts0_rsc, pts1_rsc;
      pts0_rsc.reserve(matches_good.size());
      pts1_rsc.reserve(matches_good.size());
      for (size_t i = 0; i < matches_good.size(); i++) {
        const int index_pt0 = matches_good.at(i).queryIdx;
        const int index_pt1 = matches_good.at(i).trainIdx;
        pts0_rsc.push_back(pts0[index_pt0].pt);
        pts1_rsc.push_back(pts1[index_pt1].pt);
      }

      if (pts0_rsc.size() < 10) {
        PRINT_ALL("not enough points for ransac, only %zu matches\n", pts0_rsc.size());
        return;
      }

      std::vector<cv::Point2f> pts0_n, pts1_n;
      pts0_n.reserve(pts0_rsc.size());
      pts1_n.reserve(pts1_rsc.size());
      for (size_t i = 0; i < pts0_rsc.size(); i++) {
        pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0_rsc.at(i)));
        pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1_rsc.at(i)));
      }

      std::vector<uchar> mask_rsc;
      double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
      double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
      double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
      cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 1 / max_focallength, 0.999, mask_rsc);

      if (mask_rsc.size() != matches_good.size()) {
        PRINT_ALL("RANSAC mask size does not match matches size, something went wrong\n");
        return;
      }

      for (size_t i = 0; i < matches_good.size(); i++) {
        if (mask_rsc[i] != 1) {
          continue;
        }
        matches.push_back(matches_good.at(i));
      }
      return;
    }
    PRINT_WARNING(YELLOW "[TRACK-DESC]: SuperGlue worker failed, falling back to descriptor KNN matching.\n" RESET);
    }
  }

  // Our 1to2 and 2to1 match vectors
  std::vector<std::vector<cv::DMatch>> matches0to1, matches1to0;

  // Match descriptors (return 2 nearest neighbours)
  matcher->knnMatch(desc0, desc1, matches0to1, 2);
  matcher->knnMatch(desc1, desc0, matches1to0, 2);
  PRINT_ALL("matches before ratio test = %zu\n", matches0to1.size());
  
  // Do a ratio test for both matches
  robust_ratio_test(matches0to1);
  robust_ratio_test(matches1to0);

  PRINT_ALL("matches after ratio test = %zu\n", matches0to1.size());
  // Finally do a symmetry test
  std::vector<cv::DMatch> matches_good;
  robust_symmetry_test(matches0to1, matches1to0, matches_good);
  PRINT_ALL("matches good after ratio and symmetry test = %zu\n", matches_good.size());

  // Convert points into points for RANSAC
  // Creates point pairs 
  std::vector<cv::Point2f> pts0_rsc, pts1_rsc;
  for (size_t i = 0; i < matches_good.size(); i++) {
    // Get our ids
    int index_pt0 = matches_good.at(i).queryIdx;
    int index_pt1 = matches_good.at(i).trainIdx;
    // Push back just the 2d point
    pts0_rsc.push_back(pts0[index_pt0].pt);
    pts1_rsc.push_back(pts1[index_pt1].pt);
  }

  // If we don't have enough points for ransac just return empty
  if (pts0_rsc.size() < 10) {
    PRINT_ALL("not enough points for ransac, only %zu matches\n", pts0_rsc.size());
    return;
  }

  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  std::vector<cv::Point2f> pts0_n, pts1_n;
  for (size_t i = 0; i < pts0_rsc.size(); i++) {
    pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0_rsc.at(i)));
    pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1_rsc.at(i)));
  }

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
  double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
  //for ransac it N at least 15 points
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 1 / max_focallength, 0.999, mask_rsc);

  if (mask_rsc.size() != matches_good.size()) {
    PRINT_ALL("RANSAC mask size does not match matches size, something went wrong\n");
    return;
  }

  // Loop through all good matches, and only append ones that have passed RANSAC
  for (size_t i = 0; i < matches_good.size(); i++) {
    // Skip if bad ransac id
    if (mask_rsc[i] != 1)
    //how often this happens?
      continue;
    // Else, lets append this match to the return array!
    matches.push_back(matches_good.at(i));
  }
}

void TrackDescriptor::robust_ratio_test(std::vector<std::vector<cv::DMatch>> &matches) {
  // Loop through all matches
  for (auto &match : matches) {
    // If 2 NN has been identified, else remove this feature
    if (match.size() > 1) {
      if (match[1].distance <= std::numeric_limits<float>::epsilon()) {
        match.clear();
        PRINT_ALL("Warning: 2nd nearest neighbor has zero distance, skipping ratio test for this match\n");
        continue;
      }
      // check distance ratio, remove it if the ratio is larger
      if (match[0].distance / match[1].distance > knn_ratio) {
        match.clear();
      }
    } else {
      // does not have 2 neighbours, so remove it
      match.clear();
    }
  }
}

void TrackDescriptor::robust_symmetry_test(std::vector<std::vector<cv::DMatch>> &matches1, std::vector<std::vector<cv::DMatch>> &matches2,
                                           std::vector<cv::DMatch> &good_matches) {
  // for all matches image 1 -> image 2
  for (auto &match1 : matches1) {
    // ignore deleted matches
    if (match1.empty() || match1.size() < 2)
      continue;
    // for all matches image 2 -> image 1
    for (auto &match2 : matches2) {
      // ignore deleted matches
      if (match2.empty() || match2.size() < 2)
        continue;
      // Match symmetry test
      if (match1[0].queryIdx == match2[0].trainIdx && match2[0].queryIdx == match1[0].trainIdx) {
        // add symmetrical match
        good_matches.emplace_back(cv::DMatch(match1[0].queryIdx, match1[0].trainIdx, match1[0].distance));
        // next match in image 1 -> image 2
        break;
      }
    }
  }
}
