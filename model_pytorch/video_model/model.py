import torch, math
import torch.nn as nn
import torch.nn.functional as F

from convlstm import ConvLSTM

def conv_downsample(in_ch, out_ch, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size,stride=1, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )

def conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3,padding=1),
        nn.ReLU(inplace=True)
    )

def deconv(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

def crop(input, ref): #when skip connection tensor dims are one off
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

class CNN_Single(nn.Module): #convolutional DispNet
    def __init__(self,input_dim=3):
        super(CNN_Single, self).__init__()

        #downsampling convolutions
        self.conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv0 = conv_downsample(input_dim, self.conv_planes[0], kernel_size=7)
        self.conv1 = conv_downsample(self.conv_planes[0], self.conv_planes[1], kernel_size=5)
        self.conv2 = conv_downsample(self.conv_planes[1], self.conv_planes[2])
        self.conv3 = conv_downsample(self.conv_planes[2], self.conv_planes[3])
        self.conv4 = conv_downsample(self.conv_planes[3], self.conv_planes[4])
        self.conv5 = conv_downsample(self.conv_planes[4], self.conv_planes[5])

        self.conv6 = conv_downsample(self.conv_planes[5], self.conv_planes[6])

        #upsampling convolutions
        self.upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv0 = conv(self.upconv_planes[0]+ self.conv_planes[5], self.upconv_planes[0])
        self.upconv1 = conv(self.upconv_planes[1]+ self.conv_planes[4], self.upconv_planes[1])
        self.upconv2 = conv(self.upconv_planes[2]+ self.conv_planes[3], self.upconv_planes[2])
        self.upconv3 = conv(self.upconv_planes[3]+ self.conv_planes[2], self.upconv_planes[3])

        #deconvolutions
        self.deconv0 = deconv(self.conv_planes[5], self.upconv_planes[0])
        self.deconv1 = deconv(self.upconv_planes[0], self.upconv_planes[1])
        self.deconv2 = deconv(self.upconv_planes[1], self.upconv_planes[2])
        self.deconv3 = deconv(self.upconv_planes[2], self.upconv_planes[3])
        self.deconv4 = deconv(self.upconv_planes[3], self.upconv_planes[4])
        self.deconv5 = deconv(self.upconv_planes[4] + self.conv_planes[1], self.upconv_planes[5])
        self.deconv6 = deconv(self.upconv_planes[5] + self.conv_planes[0], self.upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
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

    def forward(self, x):
        #encoder
        out_conv0 = self.conv0(x)
        # print(out_conv0.shape)
        out_conv1 = self.conv1(out_conv0)
        # print(out_conv1.shape)
        out_conv2 = self.conv2(out_conv1)
        # print(out_conv2.shape)
        out_conv3 = self.conv3(out_conv2)
        # print(out_conv3.shape)
        out_conv4 = self.conv4(out_conv3)
        # print(out_conv4.shape)
        out_conv5 = self.conv5(out_conv4)
        # print(out_conv5.shape)
        out_conv6 = self.conv6(out_conv5)
        # print(out_conv6.shape)

        #decoder
        out_deconv0 = self.deconv0(out_conv6)
        out_deconv0 = crop(out_deconv0,out_conv5)
        concat0 = torch.cat((out_deconv0,out_conv5), 1)
        out_upconv0 = self.upconv0(concat0)
        # print(out_upconv0.shape)

        out_deconv1 = self.deconv1(out_upconv0)
        out_deconv1 = crop(out_deconv1,out_conv4)
        concat1 = torch.cat((out_deconv1, out_conv4), 1)
        out_upconv1 = self.upconv1(concat1)
        # print(out_upconv1.shape)

        out_deconv2 = self.deconv2(out_upconv1)
        out_deconv2 = crop(out_deconv2,out_conv3)
        concat2 = torch.cat((out_deconv2, out_conv3), 1)
        out_upconv2 = self.upconv2(concat2)
        # print(out_upconv2.shape)

        out_deconv3 = self.deconv3(out_upconv2)
        out_deconv3 = crop(out_deconv3,out_conv2)
        concat3 = torch.cat((out_deconv3, out_conv2), 1)
        out_upconv3 = self.upconv3(concat3)
        # print(out_upconv3.shape)

        out_deconv4 = self.deconv4(out_upconv3)
        # print(out_deconv4.shape)

        out_deconv4 = crop(out_deconv4,out_conv1)
        concat4 = torch.cat((out_deconv4, out_conv1), 1)
        out_deconv5 = self.deconv5(concat4)
        # print(out_deconv5.shape)

        out_deconv5 = crop(out_deconv5,out_conv0)
        concat5 = torch.cat((out_deconv5, out_conv0), 1)
        out_deconv6 = self.deconv6(concat5)
        # print(out_deconv6.shape)

        final = torch.sum(out_deconv6,axis=1)
        final = final.view(final.shape[0],1,final.shape[1],final.shape[2])
        return final

class CNN_Stack(nn.Module): #convolutional DispNet
    def __init__(self,input_dim=3):
        super(CNN_Stack, self).__init__()

        #downsampling convolutions
        self.conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv0 = conv_downsample(input_dim, self.conv_planes[0], kernel_size=7)
        self.conv1 = conv_downsample(self.conv_planes[0], self.conv_planes[1], kernel_size=5)
        self.conv2 = conv_downsample(self.conv_planes[1], self.conv_planes[2])
        self.conv3 = conv_downsample(self.conv_planes[2], self.conv_planes[3])
        self.conv4 = conv_downsample(self.conv_planes[3], self.conv_planes[4])
        self.conv5 = conv_downsample(self.conv_planes[4], self.conv_planes[5])

        self.conv6 = conv_downsample(self.conv_planes[5], self.conv_planes[6])

        #upsampling convolutions
        self.upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv0 = conv(self.upconv_planes[0]+ self.conv_planes[5], self.upconv_planes[0])
        self.upconv1 = conv(self.upconv_planes[1]+ self.conv_planes[4], self.upconv_planes[1])
        self.upconv2 = conv(self.upconv_planes[2]+ self.conv_planes[3], self.upconv_planes[2])
        self.upconv3 = conv(self.upconv_planes[3]+ self.conv_planes[2], self.upconv_planes[3])

        #deconvolutions
        self.deconv0 = deconv(self.conv_planes[5], self.upconv_planes[0])
        self.deconv1 = deconv(self.upconv_planes[0], self.upconv_planes[1])
        self.deconv2 = deconv(self.upconv_planes[1], self.upconv_planes[2])
        self.deconv3 = deconv(self.upconv_planes[2], self.upconv_planes[3])
        self.deconv4 = deconv(self.upconv_planes[3], self.upconv_planes[4])
        self.deconv5 = deconv(self.upconv_planes[4] + self.conv_planes[1], self.upconv_planes[5])
        self.deconv6 = deconv(self.upconv_planes[5] + self.conv_planes[0], self.upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
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

    def forward(self, x):
        #encoder
        out_conv0 = self.conv0(x)
        # print(out_conv0.shape)
        out_conv1 = self.conv1(out_conv0)
        # print(out_conv1.shape)
        out_conv2 = self.conv2(out_conv1)
        # print(out_conv2.shape)
        out_conv3 = self.conv3(out_conv2)
        # print(out_conv3.shape)
        out_conv4 = self.conv4(out_conv3)
        # print(out_conv4.shape)
        out_conv5 = self.conv5(out_conv4)
        # print(out_conv5.shape)
        out_conv6 = self.conv6(out_conv5)
        # print(out_conv6.shape)

        #decoder
        out_deconv0 = self.deconv0(out_conv6)
        out_deconv0 = crop(out_deconv0,out_conv5)
        concat0 = torch.cat((out_deconv0,out_conv5), 1)
        out_upconv0 = self.upconv0(concat0)
        # print(out_upconv0.shape)

        out_deconv1 = self.deconv1(out_upconv0)
        out_deconv1 = crop(out_deconv1,out_conv4)
        concat1 = torch.cat((out_deconv1, out_conv4), 1)
        out_upconv1 = self.upconv1(concat1)
        # print(out_upconv1.shape)

        out_deconv2 = self.deconv2(out_upconv1)
        out_deconv2 = crop(out_deconv2,out_conv3)
        concat2 = torch.cat((out_deconv2, out_conv3), 1)
        out_upconv2 = self.upconv2(concat2)
        # print(out_upconv2.shape)

        out_deconv3 = self.deconv3(out_upconv2)
        out_deconv3 = crop(out_deconv3,out_conv2)
        concat3 = torch.cat((out_deconv3, out_conv2), 1)
        out_upconv3 = self.upconv3(concat3)
        # print(out_upconv3.shape)

        out_deconv4 = self.deconv4(out_upconv3)
        # print(out_deconv4.shape)

        out_deconv4 = crop(out_deconv4,out_conv1)
        concat4 = torch.cat((out_deconv4, out_conv1), 1)
        out_deconv5 = self.deconv5(concat4)
        # print(out_deconv5.shape)

        out_deconv5 = crop(out_deconv5,out_conv0)
        concat5 = torch.cat((out_deconv5, out_conv0), 1)
        out_deconv6 = self.deconv6(concat5)
        #print(out_deconv6.shape)

        #final = torch.sum(out_deconv6,axis=1)/out_deconv6.shape[1] #average in stack dimension
        final = torch.sum(out_deconv6,axis=1) #sum in 16 channel dimension
        final = final.view(final.shape[0],1,final.shape[1],final.shape[2])
        return final

class DenseSLAMNet(nn.Module): #recurrent variation of CNN-Single/Stack
    def __init__(self,input_dim=3,timespan=10):
        super(DenseSLAMNet, self).__init__()

        #downsampling convolutions
        self.conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv0 = conv_downsample(input_dim*timespan, self.conv_planes[0], kernel_size=7)
        self.conv1 = conv_downsample(self.conv_planes[0], self.conv_planes[1], kernel_size=5)
        self.conv2 = conv_downsample(self.conv_planes[1], self.conv_planes[2])
        self.conv3 = conv_downsample(self.conv_planes[2], self.conv_planes[3])
        self.conv4 = conv_downsample(self.conv_planes[3], self.conv_planes[4])
        self.conv5 = conv_downsample(self.conv_planes[4], self.conv_planes[5])

        self.conv6 = conv_downsample(self.conv_planes[5], self.conv_planes[6])

        #upsampling convolutions
        self.upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv0 = conv(self.upconv_planes[0] + self.conv_planes[5], self.upconv_planes[0])
        self.upconv1 = conv(self.upconv_planes[1] + self.conv_planes[4], self.upconv_planes[1])
        self.upconv2 = conv(self.upconv_planes[2] + self.conv_planes[3], self.upconv_planes[2])
        self.upconv3 = conv(self.upconv_planes[3] + self.conv_planes[2], self.upconv_planes[3])

        #deconvolutions
        self.deconv0 = deconv(self.conv_planes[5], self.upconv_planes[0])
        self.deconv1 = deconv(self.upconv_planes[0], self.upconv_planes[1])
        self.deconv2 = deconv(self.upconv_planes[1], self.upconv_planes[2])
        self.deconv3 = deconv(self.upconv_planes[2], self.upconv_planes[3])
        self.deconv4 = deconv(self.upconv_planes[3], self.upconv_planes[4])

        # self.recurrent1 = nn.LSTM(128 * 48 * 64, 128 * 48 * 64)
        self.recurrent1 = ConvLSTM(
            input_size = (48, 64), input_dim = 128, hidden_dim = [48, 64, 128], 
            kernel_size = (3, 3), num_layers = 3, batch_first = True, return_all_layers = True
        )

        self.deconv5 = deconv(self.upconv_planes[4] + self.conv_planes[1], self.upconv_planes[5])
        self.deconv6 = deconv(self.upconv_planes[5] + self.conv_planes[0], self.upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
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

    def forward(self, x):
        #encoder
        out_conv0 = self.conv0(x)
        # print(out_conv0.shape)
        out_conv1 = self.conv1(out_conv0)
        # print(out_conv1.shape)
        out_conv2 = self.conv2(out_conv1)
        # print(out_conv2.shape)
        out_conv3 = self.conv3(out_conv2)
        # print(out_conv3.shape)
        out_conv4 = self.conv4(out_conv3)
        # print(out_conv4.shape)
        out_conv5 = self.conv5(out_conv4)
        # print(out_conv5.shape)
        out_conv6 = self.conv6(out_conv5)
        # print(out_conv6.shape)

        #decoder
        out_deconv0 = self.deconv0(out_conv6)
        out_deconv0 = crop(out_deconv0,out_conv5)
        concat0 = torch.cat((out_deconv0,out_conv5), 1)
        out_upconv0 = self.upconv0(concat0)
        # print(out_upconv0.shape)

        out_deconv1 = self.deconv1(out_upconv0)
        out_deconv1 = crop(out_deconv1,out_conv4)
        concat1 = torch.cat((out_deconv1, out_conv4), 1)
        out_upconv1 = self.upconv1(concat1)
        # print(out_upconv1.shape)

        out_deconv2 = self.deconv2(out_upconv1)
        out_deconv2 = crop(out_deconv2,out_conv3)
        concat2 = torch.cat((out_deconv2, out_conv3), 1)
        out_upconv2 = self.upconv2(concat2)
        # print(out_upconv2.shape)

        out_deconv3 = self.deconv3(out_upconv2)
        out_deconv3 = crop(out_deconv3,out_conv2)
        concat3 = torch.cat((out_deconv3, out_conv2), 1)
        out_upconv3 = self.upconv3(concat3)
        # print(out_upconv3.shape)

        out_deconv4 = self.deconv4(out_upconv3)

        out_deconv4 = crop(out_deconv4,out_conv1)
        concat4 = torch.cat((out_deconv4, out_conv1), 1)

        concat_unsqueezed = concat4.unsqueeze(0)
        output, hidden = self.recurrent1(concat_unsqueezed)
        print("OUTPUT SHAPE {}".format(output[-1].shape))
        print("HIDDEN {}".format(hidden.shape))

        out_deconv5 = self.deconv5(concat4)
        # print(out_deconv5.shape)

        out_deconv5 = crop(out_deconv5,out_conv0)
        concat5 = torch.cat((out_deconv5, out_conv0), 1)
        out_deconv6 = self.deconv6(concat5)
        #print(out_deconv6.shape)

        #final = torch.sum(out_deconv6,axis=1)/out_deconv6.shape[1] #average in stack dimension
        final = torch.sum(out_deconv6,axis=1) #sum in 16 channel dimension
        final = final.view(final.shape[0],1,final.shape[1],final.shape[2])
        return final

