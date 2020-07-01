// tag::part1[]
#include "torch/script.h" // <1>
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;
int main(int argc, char **argv) {
// end::part1[]
  if (argc != 4) {
    std::cerr << "Call as " << argv[0] << " model.pt input.jpg output.jpg"
              << std::endl;
    return 1;
  }
  // tag::part2[]
  CImg<float> image(argv[2]); // <2>
  image = image.resize(227, 227); // <3>
  // end::part2[]
  // tag::part3[]
  auto input_ = torch::tensor(
    torch::ArrayRef<float>(image.data(), image.size())); // <1>
  auto input = input_.reshape({1, 3, image.height(),
			       image.width()}).div_(255); // <2>

  auto module = torch::jit::load(argv[1]); // <3>

  std::vector<torch::jit::IValue> inputs; // <4>
  inputs.push_back(input);
  auto output_ = module.forward(inputs).toTensor(); // <5>

  auto output = output_.contiguous().mul_(255); // <6>
// end::part3[]
// tag::part4[]
  CImg<float> out_img(output.data_ptr<float>(), output.size(2), // <4>
                      output.size(3), 1, output.size(1));
  out_img.save(argv[3]); // <5>
  return 0;
}
// end::part4[]
