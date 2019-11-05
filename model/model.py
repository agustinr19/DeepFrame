import math
import collections

import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def convt(self, in_channels, out_channels):       
    assert -2 - 2 * self.padding + self.kernel + self.output_padding == 0, \
           'The set of parameters doesn\'t double the input'
    
    module_name = "deconv{}".format(self.kernel)
    return nn.Sequential(collections.OrderedDict([
        (module_name, nn.ConvTranspose2d(in_channels,out_channels,self.kernel,
        self.stride,self.padding,self.output_padding,bias=False)),
        ('batchnorm', nn.BatchNorm2d(out_channels)),
        ('relu',      nn.ReLU(inplace=True)),
    ]))

# Decoder architectures specify four separate upsampling layers where the 
# dimensions decrease and the image size increases
class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder1 = None
        self.decoder2 = None
        self.decoder3 = None
        self.decoder4 = None

        self.decoders = None

    def forward(self, x):
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        return x

class Deconv(Decoder):

    _deconv_types = ['deconv2', 'deconv3']

    def __init__(self, deconv_type, in_channels):
        super().__init__()

        self.kernel = int(deconv_type[-1])
        self.stride = 2
        self.padding = (self.kernel - 1) // 2
        self.output_padding = self.kernel % 2

        self.decoder1 = convt(self, in_channels, in_channels // 2)
        self.decoder2 = convt(self, in_channels // 2, in_channels // 4)
        self.decoder3 = convt(self, in_channels // 4, in_channels // 8)
        self.decoder4 = convt(self, in_channels // 8, in_channels // 16)

class InterpolateLayer(nn.Module):

    def __init__(self, output_size, mode):
        super(InterpolateLayer, self).__init__()
        self.output_size = output_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, self.output_size, mode = self.mode)
        return x

class ResNet(nn.Module):
    
    # These should be reversed when used with the network
    L34_DEPTHS = (64, 64, 128, 256)[::-1]
    G34_DEPTHS = (64, 256, 512, 1024)[::-1]

    _resnet_encoders = {
        'resnet18'  : (models.resnet18, 512, L34_DEPTHS),
        'resnet34'  : (models.resnet34, 512, L34_DEPTHS),
        'resnet50'  : (models.resnet50, 2048, G34_DEPTHS),
        'resnet101' : (models.resnet101, 2048, G34_DEPTHS),
        'resnet152' : (models.resnet152, 2048, G34_DEPTHS)
    }

    def __init__(self, resnet_encoder, decoder_type, dims, output_size, pre_trained=True):
        assert resnet_encoder in self._resnet_encoders, \
               '{} is not a valid resnet model'.format(resnet_encoder)
        super().__init__()

        model_fn, encoder_out_channels, encoder_depths = self._resnet_encoders[resnet_encoder]
        pretrained_model = model_fn(pretrained=pre_trained)

        self.dims = dims

        # encoder architecture
        if len(self.dims) == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(len(self.dims), 64, 7, stride = 2, padding = 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv1.apply(weights_init)
            self.bn1.apply(weights_init)
        
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.encoder1 = pretrained_model._modules['layer1']
        self.encoder2 = pretrained_model._modules['layer2']
        self.encoder3 = pretrained_model._modules['layer3']
        self.encoder4 = pretrained_model._modules['layer4']

        del pretrained_model

        self.conv2 = nn.Conv2d(encoder_out_channels, encoder_out_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(encoder_out_channels // 2)

        # decoder architecture
        self.decoder = Deconv(decoder_type, encoder_out_channels // 2)
        self.conv3 = nn.Conv2d(encoder_out_channels // 32, 1, 3, padding = 1, bias=False)

        self.upsample = InterpolateLayer(output_size, 'bilinear')

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    # resnetX downsamples the input image by five powers of 2. We only care about the first four since 
    # upsampling happens with the last power. 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        e1 = self.relu(x)

        e2 = self.maxpool(e1)

        e2 = self.encoder1(e2)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)

        x = self.conv2(e5)
        x = self.bn2(x)

        x = self.decoder(x)

        x = self.conv3(x)
        x = self.upsample(x)

        return x
