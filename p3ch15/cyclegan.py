# This is the code from the p1ch2/3_cyclegan notebook

import torch
import torch.nn as nn

class ResNetBlock(nn.Module):

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # here we move to 0-1 input and 0-1 output
        # usually one would think about writing this differently
        # for efficiency (e.g. absorbing the 255 into the first conv
        return self.model(input * 255) / 2 + 0.5

def get_pretrained_model(model_path, map_location=None):
    netG = ResNetGenerator()
    model_data = torch.load(model_path, map_location=map_location)
    netG.load_state_dict(model_data)
    netG.eval()
    for p in netG.parameters():
        netG.requires_grad_(False)
    return netG

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Call as {} zebra_weights.pt traced_zebra_model.pt".format(sys.argv[0]))
        sys.exit(1)
    model = get_pretrained_model(sys.argv[1], map_location='cpu')
    traced_model = torch.jit.trace(model, torch.randn(1, 3, 227, 227))
    traced_model.save(sys.argv[2])

    # img = Image.open("../data/p1ch2/horse.jpg")
    # out_img.save('../data/p1ch2/zebra.jpg')

