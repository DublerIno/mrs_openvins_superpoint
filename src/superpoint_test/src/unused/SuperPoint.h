#ifndef SUPERPOINT_H
#define SUPERPOINT_H


#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <vector>
#include <fstream>
#include <unordered_map>
#include <stdexcept>


#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif


namespace ORB_SLAM2
{

struct SuperPoint : torch::nn::Module {
  SuperPoint();

  std::vector<torch::Tensor> forward(torch::Tensor x);

  void load_weights(const std::string& path);

  torch::nn::Conv2d conv1a;
  torch::nn::Conv2d conv1b;

  torch::nn::Conv2d conv2a;
  torch::nn::Conv2d conv2b;

  torch::nn::Conv2d conv3a;
  torch::nn::Conv2d conv3b;

  torch::nn::Conv2d conv4a;
  torch::nn::Conv2d conv4b;

  torch::nn::Conv2d convPa;
  torch::nn::Conv2d convPb;

  // descriptor
  torch::nn::Conv2d convDa;
  torch::nn::Conv2d convDb;

};

