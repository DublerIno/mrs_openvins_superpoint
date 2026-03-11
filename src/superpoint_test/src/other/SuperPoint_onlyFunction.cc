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

  

//Funtctions for 
void NMS(cv::Mat det, cv::Mat conf, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height);
void NMS2(std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint>& pts,
            int border, int dist_thresh, int img_width, int img_height);

/*
Full pipeline function that computes keypoints and descriptors from input image.
    Input
      model: SuperPoint pytorch model.
      img: Grayscale input image cv::Mat shaped H x W.
      threshold: Confidence threshold to filter points.
      nms: Whether to apply non maximum suppression to points.
      cuda: Whether to use cuda gpu acceleration.
    Output
        cv::Mat with: 
            keypoints: Vector of cv::KeyPoint containing detected points.
            descriptors: cv::Mat shaped N x 256 containing descriptors for detected points.
*/
cv::Mat SPdetect(std::shared_ptr<SuperPoint> model, cv::Mat img, std::vector<cv::KeyPoint> &keypoints, double threshold, bool nms, bool cuda)
{
    // Convert image to tensor
    //from_blob does not own the data, so we clone the image to make sure the data is contiguous and owned
    auto x = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);
    x = x.to(torch::kFloat) / 255;

    //Device selection:
    bool use_cuda = cuda && torch::cuda::is_available();
    torch::DeviceType device_type;
    if (use_cuda)
        device_type = torch::kCUDA;
    else
        device_type = torch::kCPU;
    torch::Device device(device_type);

    //moving model every frame?
    model->to(device);
    //compute forward pass
    x = x.set_requires_grad(false);
    auto out = model->forward(x.to(device));
    auto prob = out[0].squeeze(0);  // [H, W]
    auto desc = out[1];             // [1, 256, H/8, W/8]

    // Extract keypoints from heatmap by set threshold
    auto kpts = (prob > threshold);
    kpts = torch::nonzero(kpts);  // [n_keypoints, 2]  (y, x)
    auto fkpts = kpts.to(torch::kFloat);
    
    //desc is lowres, so bilinear interpolation to get desc at keypoint locations
    auto grid = torch::zeros({1, 1, kpts.size(0), 2}).to(device);  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / prob.size(1) - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / prob.size(0) - 1;  // y

    // mode=0 (bilinear), padding_mode=0 (zeros), align_corners=false
    desc = torch::grid_sampler(desc, grid, 0, 0,false);  // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);  // [256, n_keypoints]

    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));
    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]

    //move from gpu to cpu if needed
    if (use_cuda)
        desc = desc.to(torch::kCPU);

    // desc = [N of detected keypoints, 256 dim]
    // create opencv::Mat from desc tensor
    cv::Mat descriptors_no_nms(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data<float>());
    
    //copy keypoints to vector
    std::vector<cv::KeyPoint> keypoints_no_nms;
    for (int i = 0; i < kpts.size(0); i++) {
        float response = prob[kpts[i][0]][kpts[i][1]].item<float>();
        keypoints_no_nms.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1, response));
    }


    //NMS removes similar/close keypoint based on their confidence scores - Non Maximum Suppression
    if (nms) {
        cv::Mat kpt_mat(keypoints_no_nms.size(), 2, CV_32F);
        cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
        for (size_t i = 0; i < keypoints_no_nms.size(); i++) {
            int x = keypoints_no_nms[i].pt.x;
            int y = keypoints_no_nms[i].pt.y;
            kpt_mat.at<float>(i, 0) = (float)keypoints_no_nms[i].pt.x;
            kpt_mat.at<float>(i, 1) = (float)keypoints_no_nms[i].pt.y;

            conf.at<float>(i, 0) = prob[y][x].item<float>();
        }

        cv::Mat descriptors;

        int border = 8;
        int dist_thresh = 4;
        int height = img.rows;
        int width = img.cols;


        NMS(kpt_mat, conf, descriptors_no_nms, keypoints, descriptors, border, dist_thresh, width, height);

        return descriptors;
    }
    else {
        keypoints = keypoints_no_nms;
        return descriptors_no_nms.clone();
    }

    // return descriptors.clone();
}
} //namespace ORB_SLAM2