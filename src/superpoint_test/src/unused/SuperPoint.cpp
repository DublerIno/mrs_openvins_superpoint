#include "SuperPoint.h"


//using orb_slam2 namespace? = propably for easier implementation with whole slam system
namespace ORB_SLAM2
{

const int c1 = 64;
const int c2 = 64;
const int c3 = 128;
const int c4 = 128;
const int c5 = 256;
const int d1 = 256;


//SuperPoint Network Constructor
SuperPoint::SuperPoint()
      : 
        //Shared Encoder = low dimensional feature extractor 
        //module holder constructors
        conv1a(torch::nn::Conv2dOptions( 1, c1, 3).stride(1).padding(1)),
        conv1b(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1)),

        conv2a(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1)),
        conv2b(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1)),

        conv3a(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1)),
        conv3b(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1)),

        conv4a(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1)),
        conv4b(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1)),

        //Detection head - interesting points detector 
        convPa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
        convPb(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0)),

        //Descriptor Head - encodes point areas into descriptor space   
        convDa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
        convDb(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0))
        
  {
    //in cpp we need to register modules manually
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

/*
 Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
*/

std::vector<torch::Tensor> SuperPoint::forward(torch::Tensor x) {

    // Shared Encoder - VGG like blocks
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
    //B x 128 x H/8 x W/8

    // detection head
    auto cPa = torch::relu(convPa->forward(x));
    auto semi = convPb->forward(cPa);  // [B, 65, H/8, W/8]

    // descriptor head 
    auto cDa = torch::relu(convDa->forward(x));
    auto desc = convDb->forward(cDa);  // [B, d1, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    //softmax over 65 classes
    semi = torch::softmax(semi, 1);
    semi = semi.slice(1, 0, 64);
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]

    //semi to probability heatmap
    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]

    //return tensor descirptors and point heatmap
    //list of two template argument - torch::Tensor
    std::vector<torch::Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);

    return ret;
  }
 void SuperPoint::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file for reading");

    std::unordered_map<std::string, torch::Tensor> params_map;
    for (auto& pair : named_parameters()) {
        params_map[pair.key()] = pair.value();
    }

    std::unordered_map<std::string, torch::Tensor> buffers_map;
    for (auto& pair : named_buffers()) {
        buffers_map[pair.key()] = pair.value();
    }

    while (file.peek() != EOF) {
        int64_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);

        torch::Tensor* target_tensor = nullptr;
        auto it_param = params_map.find(name);
        if (it_param != params_map.end()) {
            target_tensor = &it_param->second;
        }
        else {
            auto it_buf = buffers_map.find(name);
            if (it_buf != buffers_map.end()) {
                target_tensor = &it_buf->second;
            }
            else {
                throw std::runtime_error("Parameter or buffer " + name + " not found in model");
            }
        }

        torch::Tensor& param = *target_tensor;

        int64_t ndims;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
        std::vector<int64_t> shape(ndims);
        for (int i = 0; i < ndims; ++i) {
            file.read(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
        }

        auto expected_shape = param.sizes();
        if (shape != expected_shape) {
            throw std::runtime_error("Shape mismatch for " + name);
        }

        int64_t num_elems = 1;
        for (auto dim : shape) num_elems *= dim;
        file.read(reinterpret_cast<char*>(param.data_ptr()), num_elems * param.element_size());
    }
}

  
