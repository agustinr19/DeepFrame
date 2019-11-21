import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

def conv_downsample(in_ch, out_ch, kernel_size=3):
    return nn.Sequential(
    	nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size),
        nn.ReLU(inplace=True)
    )

def conv_upsample(in_ch, out_ch):
	return nn.Sequential(
    	nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

def conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def match_size(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class CNN_SINGLE(nn.Module): #convolutional DispNet on single frame data
    def __init__(self,frame_dim):
        super(CNN_SINGLE, self).__init__()

        #downsampling convolutions
	    conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = conv_downsample(3, conv_planes[0], kernel_size=7)
        self.conv2 = conv_downsample(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv_downsample(conv_planes[1], conv_planes[2])
        self.conv4 = conv_downsample(conv_planes[2], conv_planes[3])
        self.conv5 = conv_downsample(conv_planes[3], conv_planes[4])
        self.conv6 = conv_downsample(conv_planes[4], conv_planes[5])
        self.conv7 = conv_downsample(conv_planes[5], conv_planes[6])

        #upsampling convolutions
        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv1 = conv_upsample(conv_planes[6],   upconv_planes[0])
        self.upconv2 = conv_upsample(upconv_planes[0], upconv_planes[1])
        self.upconv3 = conv_upsample(upconv_planes[1], upconv_planes[2])
        self.upconv4 = conv_upsample(upconv_planes[2], upconv_planes[3])
        self.upconv5 = conv_upsample(upconv_planes[3], upconv_planes[4])
        self.upconv6 = conv_upsample(upconv_planes[4], upconv_planes[5])
        self.upconv7 = conv_upsample(upconv_planes[5], upconv_planes[6])

        #deconvolutions
        self.deconv1 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.deconv2 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.deconv3 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.deconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.deconv5 = conv(upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.deconv6 = conv(upconv_planes[5] + conv_planes[0], upconv_planes[5])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
    	
    	#encoder
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        #decoder
        out_upconv1 = match_size(self.upconv1(out_conv7), out_conv6)
        
        concat1 = torch.cat((out_upconv1, out_conv6), 1)
        out_deconv1 = self.deconv1(concat1)
        out_upconv2 = match_size(self.upconv2(out_deconv1), out_conv5)
        
        concat2 = torch.cat((out_upconv2, out_conv5), 1)
        out_deconv2 = self.deconv2(concat2)
        out_upconv3 = match_size(self.upconv3(out_deconv2), out_conv4)

        concat3 = torch.cat((out_upconv3, out_conv4), 1)
        out_deconv3 = self.deconv3(concat3)
        out_upconv4 = match_size(self.upconv4(out_deconv3), out_conv3)

        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_deconv4 = self.deconv4(concat4)
        out_upconv5 = match_size(self.upconv5(out_deconv4), out_conv2)

        concat5 = torch.cat((out_upconv5, out_conv2), 1)
        out_deconv5 = self.deconv5(concat5)
        out_upconv6 = match_size(self.upconv6(out_deconv5), out_conv1)

        concat6 = torch.cat((out_upconv6, out_conv1), 1) 
        out_deconv6 = self.deconv6(concat6)
        out_upconv7 = match_size(self.upconv7(out_deconv6), x)

        return out_upconv7
