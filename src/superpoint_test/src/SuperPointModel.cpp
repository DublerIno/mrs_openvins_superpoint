#include "SuperPointModel.h"

namespace ORB_SLAM3 {

namespace {

constexpr int kC1 = 64;
constexpr int kC2 = 64;
constexpr int kC3 = 128;
constexpr int kC4 = 128;
constexpr int kC5 = 256;
constexpr int kD1 = 256;

}  // namespace

SuperPoint::SuperPoint()
    : conv1a(torch::nn::Conv2dOptions(1, kC1, 3).stride(1).padding(1)),
      conv1b(torch::nn::Conv2dOptions(kC1, kC1, 3).stride(1).padding(1)),
      conv2a(torch::nn::Conv2dOptions(kC1, kC2, 3).stride(1).padding(1)),
      conv2b(torch::nn::Conv2dOptions(kC2, kC2, 3).stride(1).padding(1)),
      conv3a(torch::nn::Conv2dOptions(kC2, kC3, 3).stride(1).padding(1)),
      conv3b(torch::nn::Conv2dOptions(kC3, kC3, 3).stride(1).padding(1)),
      conv4a(torch::nn::Conv2dOptions(kC3, kC4, 3).stride(1).padding(1)),
      conv4b(torch::nn::Conv2dOptions(kC4, kC4, 3).stride(1).padding(1)),
      convPa(torch::nn::Conv2dOptions(kC4, kC5, 3).stride(1).padding(1)),
      convPb(torch::nn::Conv2dOptions(kC5, 65, 1).stride(1).padding(0)),
      convDa(torch::nn::Conv2dOptions(kC4, kC5, 3).stride(1).padding(1)),
      convDb(torch::nn::Conv2dOptions(kC5, kD1, 1).stride(1).padding(0)) {
  register_module("conv1a", conv1a);
  register_module("conv1b", conv1b);
  register_module("conv2a", conv2a);
  register_module("conv2b", conv2b);
  register_module("conv3a", conv3a);
  register_module("conv3b", conv3b);
  register_module("conv4a", conv4a);
  register_module("conv4b", conv4b);
  register_module("convPa", convPa);
  register_module("convPb", convPb);
  register_module("convDa", convDa);
  register_module("convDb", convDb);
}

std::vector<torch::Tensor> SuperPoint::forward(torch::Tensor x) {
  x = torch::relu(conv1a->forward(x));
  x = torch::relu(conv1b->forward(x));
  x = torch::max_pool2d(x, 2, 2);

  x = torch::relu(conv2a->forward(x));
  x = torch::relu(conv2b->forward(x));
  x = torch::max_pool2d(x, 2, 2);

  x = torch::relu(conv3a->forward(x));
  x = torch::relu(conv3b->forward(x));
  x = torch::max_pool2d(x, 2, 2);

  x = torch::relu(conv4a->forward(x));
  x = torch::relu(conv4b->forward(x));

  auto cPa = torch::relu(convPa->forward(x));
  auto semi = convPb->forward(cPa);

  auto cDa = torch::relu(convDa->forward(x));
  auto desc = convDb->forward(cDa);

  desc = desc.div(torch::norm(desc, 2, 1, true).clamp_min(1e-8));

  semi = torch::softmax(semi, 1);
  semi = semi.slice(1, 0, 64);
  semi = semi.permute({0, 2, 3, 1}).contiguous();

  const int64_t height_cells = semi.size(1);
  const int64_t width_cells = semi.size(2);
  semi = semi.view({-1, height_cells, width_cells, 8, 8});
  semi = semi.permute({0, 1, 3, 2, 4}).contiguous();
  semi = semi.view({-1, height_cells * 8, width_cells * 8});

  return {semi, desc};
}

void SuperPoint::load_weights(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open SuperPoint weights: " + path);
  }

  std::unordered_map<std::string, torch::Tensor> params_map;
  for (auto& pair : named_parameters()) {
    params_map[pair.key()] = pair.value();
  }

  std::unordered_map<std::string, torch::Tensor> buffers_map;
  for (auto& pair : named_buffers()) {
    buffers_map[pair.key()] = pair.value();
  }

  while (file.peek() != EOF) {
    int64_t name_len = 0;
    file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    if (!file) {
      throw std::runtime_error("Failed reading parameter name length from: " + path);
    }

    std::string name(static_cast<size_t>(name_len), '\0');
    file.read(name.data(), name_len);
    if (!file) {
      throw std::runtime_error("Failed reading parameter name from: " + path);
    }

    torch::Tensor* target_tensor = nullptr;
    if (auto it = params_map.find(name); it != params_map.end()) {
      target_tensor = &it->second;
    } else if (auto it = buffers_map.find(name); it != buffers_map.end()) {
      target_tensor = &it->second;
    } else {
      throw std::runtime_error("Parameter or buffer not found in model: " + name);
    }

    int64_t ndims = 0;
    file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
    if (!file) {
      throw std::runtime_error("Failed reading tensor rank for: " + name);
    }

    std::vector<int64_t> shape(static_cast<size_t>(ndims));
    for (int64_t i = 0; i < ndims; ++i) {
      file.read(reinterpret_cast<char*>(&shape[static_cast<size_t>(i)]), sizeof(int64_t));
      if (!file) {
        throw std::runtime_error("Failed reading tensor shape for: " + name);
      }
    }

    torch::Tensor& param = *target_tensor;
    if (shape != param.sizes().vec()) {
      throw std::runtime_error("Shape mismatch for " + name);
    }

    int64_t num_elems = 1;
    for (const int64_t dim : shape) {
      num_elems *= dim;
    }

    file.read(reinterpret_cast<char*>(param.data_ptr()), num_elems * param.element_size());
    if (!file) {
      throw std::runtime_error("Failed reading tensor data for: " + name);
    }
  }
}

}  // namespace ORB_SLAM3
