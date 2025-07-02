import torch
import torch.nn as nn

from torch.nn import Module, Conv2d, Parameter, Softmax
# We would like to thank the authors of TwinLiteNet for their repository containing the different classes used in this work.
# Link to work: https://github.com/chequanghuy/TwinLiteNet
# -----------------------
class PAM_Module(Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
# ---------------
###
class UPx2(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

    def fuseforward(self, input):
        output = self.deconv(input)
        output = self.act(output)
        return output


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        # self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        # self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

    def fuseforward(self, input):
        output = self.conv(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output





class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output



class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.nOut = nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        # print("bf bn :",input.size(),self.nOut)
        output = self.bn(input)
        # print("after bn :",output.size())
        output = self.act(output)
        # print("after act :",output.size())
        return output

class ESP_RESNXT(nn.Module):
    def __init__(self, nIn, nOut, fm = 5,stride=1, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param fm: number of feature maps before dilation
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        
        # Divide Feature Maps
        fm = max(int(nIn/3),5)
        
        n = max(int(fm / 5), 1)
        n1 = max(fm - 4 * n, 1)
        
        self.C1 = C(nIn, fm, 1,stride)
        self.C2 = C(nIn, fm, 1,stride)
        self.C3 = C(nIn, fm, 1,stride)
        
        self.residual = C(nIn, nOut, 1, stride)
        
        # Set 1
        self.d11 = CDilated(fm, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d21 = CDilated(fm, n, 3, 1, 2)  # dilation rate of 2^1
        self.d41 = CDilated(fm, n, 3, 1, 4)  # dilation rate of 2^2
        self.d81 = CDilated(fm, n, 3, 1, 8)  # dilation rate of 2^3
        self.d161 = CDilated(fm, n, 3, 1, 16)  # dilation rate of 2^4
        self.C1_1 = C(fm, nOut, 1, 1)
        
        # Set 2
        self.d12 = CDilated(fm, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d22 = CDilated(fm, n, 3, 1, 2)  # dilation rate of 2^1
        self.d42 = CDilated(fm, n, 3, 1, 4)  # dilation rate of 2^2
        self.d82 = CDilated(fm, n, 3, 1, 8)  # dilation rate of 2^3
        self.d162 = CDilated(fm, n, 3, 1, 16)  # dilation rate of 2^4
        self.C2_1 = C(fm, nOut, 1,  1)
        
        # Set 3
        self.d13 = CDilated(fm, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d23 = CDilated(fm, n, 3, 1, 2)  # dilation rate of 2^1
        self.d43 = CDilated(fm, n, 3, 1, 4)  # dilation rate of 2^2
        self.d83 = CDilated(fm, n, 3, 1, 8)  # dilation rate of 2^3
        self.d163 = CDilated(fm, n, 3, 1, 16)  # dilation rate of 2^4
        self.C3_1 = C(fm, nOut, 1,  1)


        self.bn = BR(nOut)
        self.add = add
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # parallel branches
        output1 = self.C1(input)
        output2 = self.C2(input)
        output3 = self.C3(input)
        
        # Split Output1
        d11 = self.d11(output1)
        d21 = self.d21(output1)
        d41 = self.d41(output1)
        d81 = self.d81(output1)
        d161 = self.d161(output1)
        
        # heirarchical fusion for de-gridding
        add11 = d21
        add21 = add11 + d41
        add31 = add21 + d81
        add41 = add31 + d161
        combine1 = torch.cat([d11, add11, add21, add31, add41], 1)
        combine1 = self.C1_1(combine1)
        
        # Split Output2
        d12 = self.d12(output2)
        d22 = self.d22(output2)
        d42 = self.d42(output2)
        d82 = self.d82(output2)
        d162 = self.d162(output2)
        
        # heirarchical fusion for de-gridding
        add12 = d22
        add22 = add12 + d42
        add32 = add22 + d82
        add42 = add32 + d162
        combine2 = torch.cat([d12, add12, add22, add32, add42], 1)
        combine2 = self.C2_1(combine2)
        
        # Split Output3
        d13 = self.d13(output3)
        d23 = self.d23(output3)
        d43 = self.d43(output3)
        d83 = self.d83(output3)
        d163 = self.d163(output3)
        
        # heirarchical fusion for de-gridding
        add13 = d23
        add23 = add13 + d43
        add33 = add23 + d83
        add43 = add33 + d163
        combine3 = torch.cat([d13, add13, add23, add33, add43], 1)
        combine3 = self.C3_1(combine3)

        if self.add:
            combine = self.residual(input) + (combine1+combine2+combine3)

        output = self.bn(combine)
        return output



class Encoder(nn.Module):
    '''
    This class defines the LandSeg Encoder in the paper
    '''

    def __init__(self):
        # def __init__(self, classes=20, p=1, q=1):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()

        # Individual Branches
        self.RGB_level1 = ESP_RESNXT(3, 15, fm = 15,stride=2, add=True)
        self.RGB_level2 = ESP_RESNXT(15, 30, fm = 15,stride=2, add=True)
        self.RGB_level3 = ESP_RESNXT(30, 60, fm = 15,stride=2, add=True)
        self.RGB_level4 = ESP_RESNXT(60, 120, fm = 15,stride=1, add=True)
        
        
        self.DEPTH_level1 = ESP_RESNXT(1, 15, fm = 15,stride=2, add=True)
        self.DEPTH_level2 = ESP_RESNXT(15, 30, fm = 15,stride=2, add=True)
        self.DEPTH_level3 = ESP_RESNXT(30, 60, fm = 15,stride=2, add=True)
        self.DEPTH_level4 = ESP_RESNXT(60, 120, fm = 15,stride=1, add=True)

        self.fusion_level1 = ESP_RESNXT(30, 60, fm = 15,stride=2, add=True)
        self.fusion_level2 = ESP_RESNXT(60, 120, fm = 15,stride=1, add=True)
        self.fusion_level3 = ESP_RESNXT(120, 240, fm = 15,stride=2, add=True)
        self.fusion_level3_PAM = PAM_Module(240)
        self.fusion_level3_CAM = CAM_Module(240)

        self.reduce = CBR(240, 120, 3, 1)
        self.classify = CBR(120, 60, 1, 1)

    def forward(self, RGB_input, Depth_input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        
        output_RGB_15 = self.RGB_level1(RGB_input)
        output_RGB_30 = self.RGB_level2(output_RGB_15)
        output_RGB_60 = self.RGB_level3(output_RGB_30)
        output_RGB_120 = self.RGB_level4(output_RGB_60)
        
        output_DEPTH_15 = self.DEPTH_level1(Depth_input)
        output_DEPTH_30 = self.DEPTH_level2(output_DEPTH_15)
        output_DEPTH_60 = self.DEPTH_level3(output_DEPTH_30)
        output_DEPTH_120 = self.DEPTH_level4(output_DEPTH_60)

        output_fuse_30 = output_RGB_30 + output_DEPTH_30
        output_fuse_60 = self.fusion_level1(output_fuse_30)      
        output_fuse_120 = self.fusion_level2(output_fuse_60 + output_RGB_60 + output_DEPTH_60)
        output_fuse_240 = self.fusion_level3(output_fuse_120 + output_RGB_120 + output_DEPTH_120)
        output_fuse_240_PAM= self.fusion_level3_PAM(output_fuse_240)
        output_fuse_240_CAM= self.fusion_level3_CAM(output_fuse_240)
        output_fuse_240_DAM = output_fuse_240_PAM+output_fuse_240_CAM

        output_fuse_120_reduce = self.reduce(output_fuse_240_DAM)
        output_fuse_60_class = self.classify(output_fuse_120_reduce)

        return output_fuse_60_class


class LandSeg(nn.Module):
    '''
    Landing Surface Segmentation
    '''

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.up_1_1 = UPx2(60, 30)
        self.up_1_2 = UPx2(30, 15)
        self.up_1_3 = UPx2(15, 10)
        self.classLS = UPx2(10, 2) # Classify Landing Surface

    def forward(self, RGB_input,Depth_input):
        x = self.encoder(RGB_input,Depth_input)
        x11 = self.up_1_1(x)
        x12 = self.up_1_2(x11)
        x13 = self.up_1_3(x12)
        out=self.classLS(x13)
        return (out)

