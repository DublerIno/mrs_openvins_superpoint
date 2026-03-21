#ifndef SUPERPOINTMODEL_H
#define SUPERPOINTMODEL_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace ORB_SLAM3 {

struct SuperPoint : torch::nn::Module {
  SuperPoint();

  std::vector<torch::Tensor> forward(torch::Tensor x);
  void load_weights(const std::string& path);

  torch::nn::Conv2d conv1a{nullptr};
  torch::nn::Conv2d conv1b{nullptr};
  torch::nn::Conv2d conv2a{nullptr};
  torch::nn::Conv2d conv2b{nullptr};
  torch::nn::Conv2d conv3a{nullptr};
  torch::nn::Conv2d conv3b{nullptr};
  torch::nn::Conv2d conv4a{nullptr};
  torch::nn::Conv2d conv4b{nullptr};
  torch::nn::Conv2d convPa{nullptr};
  torch::nn::Conv2d convPb{nullptr};
  torch::nn::Conv2d convDa{nullptr};
  torch::nn::Conv2d convDb{nullptr};
};

}  // namespace ORB_SLAM3

#endif  // SUPERPOINTMODEL_H
