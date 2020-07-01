// tag::header[]
#include <torch/torch.h>  // <1>
#define cimg_use_jpeg
#include <CImg.h>
using torch::Tensor;  // <2>
// end::header[]

// at the time of writing this code (shortly after PyTorch 1.3),
// the C++ api wasn't complete and (in the case of ReLU) bug-free,
// so we define some Modules ad-hoc here.
// Chances are, that you can take standard models if and when
// they are done.

struct ConvTranspose2d : torch::nn::Module {
  // we don't do any of the running stats business
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> output_padding_;
  std::vector<int64_t> dilation_;
  Tensor weight;
  Tensor bias;

  ConvTranspose2d(int64_t in_channels, int64_t out_channels,
                  int64_t kernel_size, int64_t stride, int64_t padding,
                  int64_t output_padding)
      : stride_(2, stride), padding_(2, padding),
        output_padding_(2, output_padding), dilation_(2, 1) {
    // not good init...
    weight = register_parameter(
        "weight",
        torch::randn({out_channels, in_channels, kernel_size, kernel_size}));
    bias = register_parameter("bias", torch::randn({out_channels}));
  }
  Tensor forward(const Tensor &inp) {
    return conv_transpose2d(inp, weight, bias, stride_, padding_,
                            output_padding_, /*groups=*/1, dilation_);
  }
};

// tag::block[]
struct ResNetBlock : torch::nn::Module {
  torch::nn::Sequential conv_block;
  ResNetBlock(int64_t dim)
      : conv_block(  // <1>
            torch::nn::ReflectionPad2d(1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)),
            torch::nn::InstanceNorm2d(
	       torch::nn::InstanceNorm2dOptions(dim)),
            torch::nn::ReLU(/*inplace=*/true),
	    torch::nn::ReflectionPad2d(1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)),
            torch::nn::InstanceNorm2d(
	       torch::nn::InstanceNorm2dOptions(dim))) {
    register_module("conv_block", conv_block); // <2>
  }

  Tensor forward(const Tensor &inp) {
    return inp + conv_block->forward(inp); // <3>
  }
};
// end::block[]

// tag::generator1[]
struct ResNetGeneratorImpl : torch::nn::Module {
  torch::nn::Sequential model;
  ResNetGeneratorImpl(int64_t input_nc = 3, int64_t output_nc = 3,
                      int64_t ngf = 64, int64_t n_blocks = 9) {
    TORCH_CHECK(n_blocks >= 0);
    model->push_back(torch::nn::ReflectionPad2d(3)); // <1>
// end::generator1[]
    model->push_back(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_nc, ngf, 7)));
    model->push_back(
        torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(7)));
    model->push_back(torch::nn::ReLU(/*inplace=*/true));
    constexpr int64_t n_downsampling = 2;

    for (int64_t i = 0; i < n_downsampling; i++) {
      int64_t mult = 1 << i;
      // tag::generator2[]
      model->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(ngf * mult, ngf * mult * 2, 3)
              .stride(2)
              .padding(1))); // <3>
      // end::generator2[]
      model->push_back(torch::nn::InstanceNorm2d(
          torch::nn::InstanceNorm2dOptions(ngf * mult * 2)));
      model->push_back(torch::nn::ReLU(/*inplace=*/true));
    }

    int64_t mult = 1 << n_downsampling;
    for (int64_t i = 0; i < n_blocks; i++) {
      model->push_back(ResNetBlock(ngf * mult));
    }
    for (int64_t i = 0; i < n_downsampling; i++) {
      int64_t mult = 1 << (n_downsampling - i);
      model->push_back(
          ConvTranspose2d(ngf * mult, ngf * mult / 2, /*kernel_size=*/3,
                          /*stride=*/2, /*padding=*/1, /*output_padding=*/1));
      model->push_back(torch::nn::InstanceNorm2d(
          torch::nn::InstanceNorm2dOptions((ngf * mult / 2))));
      model->push_back(torch::nn::ReLU(/*inplace=*/true));
    }
    model->push_back(torch::nn::ReflectionPad2d(3));
    model->push_back(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf, output_nc, 7)));
    model->push_back(torch::nn::Tanh());
    // tag::generator3[]
    register_module("model", model);
  }
  Tensor forward(const Tensor &inp) { return model->forward(inp); }
};

TORCH_MODULE(ResNetGenerator); // <4>
// end::generator3[]

int main(int argc, char **argv) {
  // tag::main1[]
  ResNetGenerator model; // <1>
  // end::main1[]
  if (argc != 3) {
    std::cerr << "call as " << argv[0] << " model_weights.pt image.jpg"
              << std::endl;
    return 1;
  }
  // tag::main2[]
  torch::load(model, argv[1]); // <2>
  // end::main2[]
  // you can print the model structure just like you would in PyTorch
  // std::cout << model << std::endl;
  // tag::main3[]
  cimg_library::CImg<float> image(argv[2]);
  image.resize(400, 400);
  auto input_ =
      torch::tensor(torch::ArrayRef<float>(image.data(), image.size()));
  auto input = input_.reshape({1, 3, image.height(), image.width()});
  torch::NoGradGuard no_grad;          // <3>
  
  model->eval();                       // <4>
  
  auto output = model->forward(input); // <5>
  // end::main3[]
  // tag::main4[]
  cimg_library::CImg<float> out_img(output.data_ptr<float>(),
				    output.size(3), output.size(2),
				    1, output.size(1));
  cimg_library::CImgDisplay disp(out_img, "See a C++ API zebra!"); // <6>
  while (!disp.is_closed()) {
    disp.wait();
  }
  // end::main4[]
  return 0;
}
