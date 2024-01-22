from torch import nn as nn
from torch.nn import functional as F
import torch

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer


class F_ext(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(F_ext, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out

@ARCH_REGISTRY.register()
class SRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(SRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        out = self.conv_last(self.lrelu(self.conv_hr(out)))

        out += x
        return out

@ARCH_REGISTRY.register()
class SRResNet_EP(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, prompt_nf=64,num_block=16, upscale=4):
        super(SRResNet_EP, self).__init__()

        self.base_nf = num_feat
        self.out_nc = num_out_ch

        self.F_ext_net = F_ext(in_nc=1, nf=64)

        self.prompt_scale1 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale2 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale3 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale4 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale5 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale6 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale7 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale8 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale9 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale10 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale11 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale12 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale13 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale14 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale15 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale16 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scalehr = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scaleout = nn.Linear(prompt_nf, 3, bias=True)

        self.prompt_shift1 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift2 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift3 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift4 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift5 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift6 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift7 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift8 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift9 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift10 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift11 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift12 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift13 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift14 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift15 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift16 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shifthr = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shiftout = nn.Linear(prompt_nf, 3, bias=True)


        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):


        prompt = torch.unsqueeze(x[1], 1)

        prompt = self.F_ext_net(prompt)


        scale1 = self.prompt_scale1(prompt)
        shift1 = self.prompt_shift1(prompt)

        scale2 = self.prompt_scale2(prompt)
        shift2 = self.prompt_shift2(prompt)

        scale3 = self.prompt_scale3(prompt)
        shift3 = self.prompt_shift3(prompt)

        scale4 = self.prompt_scale4(prompt)
        shift4 = self.prompt_shift4(prompt)

        scale5 = self.prompt_scale5(prompt)
        shift5 = self.prompt_shift5(prompt)

        scale6 = self.prompt_scale6(prompt)
        shift6 = self.prompt_shift6(prompt)

        scale7 = self.prompt_scale7(prompt)
        shift7 = self.prompt_shift7(prompt)

        scale8 = self.prompt_scale8(prompt)
        shift8 = self.prompt_shift8(prompt)

        scale9 = self.prompt_scale9(prompt)
        shift9 = self.prompt_shift9(prompt)

        scale10 = self.prompt_scale10(prompt)
        shift10 = self.prompt_shift10(prompt)

        scale11 = self.prompt_scale11(prompt)
        shift11 = self.prompt_shift11(prompt)

        scale12 = self.prompt_scale12(prompt)
        shift12 = self.prompt_shift12(prompt)

        scale13 = self.prompt_scale13(prompt)
        shift13 = self.prompt_shift13(prompt)

        scale14 = self.prompt_scale14(prompt)
        shift14 = self.prompt_shift14(prompt)

        scale15 = self.prompt_scale15(prompt)
        shift15 = self.prompt_shift15(prompt)

        scale16 = self.prompt_scale16(prompt)
        shift16 = self.prompt_shift16(prompt)

        scalehr = self.prompt_scalehr(prompt)
        shifthr = self.prompt_shifthr(prompt)

        scaleout = self.prompt_scaleout(prompt)
        shiftout = self.prompt_shiftout(prompt)


        feat = self.lrelu(self.conv_first(x[0]))

        feat = self.body1(feat)
        feat = feat * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body2(feat)
        feat = feat * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body3(feat)
        feat = feat * scale3.view(-1, self.base_nf, 1, 1) + shift3.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body4(feat)
        feat = feat * scale4.view(-1, self.base_nf, 1, 1) + shift4.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body5(feat)
        feat = feat * scale5.view(-1, self.base_nf, 1, 1) + shift5.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body6(feat)
        feat = feat * scale6.view(-1, self.base_nf, 1, 1) + shift6.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body7(feat)
        feat = feat * scale7.view(-1, self.base_nf, 1, 1) + shift7.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body8(feat)
        feat = feat * scale8.view(-1, self.base_nf, 1, 1) + shift8.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body9(feat)
        feat = feat * scale9.view(-1, self.base_nf, 1, 1) + shift9.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body10(feat)
        feat = feat * scale10.view(-1, self.base_nf, 1, 1) + shift10.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body11(feat)
        feat = feat * scale11.view(-1, self.base_nf, 1, 1) + shift11.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body12(feat)
        feat = feat * scale12.view(-1, self.base_nf, 1, 1) + shift12.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body13(feat)
        feat = feat * scale13.view(-1, self.base_nf, 1, 1) + shift13.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body14(feat)
        feat = feat * scale14.view(-1, self.base_nf, 1, 1) + shift14.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body15(feat)
        feat = feat * scale15.view(-1, self.base_nf, 1, 1) + shift15.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body16(feat)
        feat = feat * scale16.view(-1, self.base_nf, 1, 1) + shift16.view(-1, self.base_nf, 1, 1) + feat


        feat = self.lrelu(self.conv_hr(feat))
        feat = feat * scalehr.view(-1, self.base_nf, 1, 1) + shifthr.view(-1, self.base_nf, 1, 1) + feat

        feat = self.conv_last(feat)
        out = feat * scaleout.view(-1, 3, 1, 1) + shiftout.view(-1, 3, 1, 1) + feat
        base = F.interpolate(x[0], scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

@ARCH_REGISTRY.register()
class SRResNet_AP(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, prompt_nf=64,num_block=16, upscale=4):
        super(SRResNet_AP, self).__init__()

        self.base_nf = num_feat
        self.out_nc = num_out_ch

        self.F_ext_net = F_ext(in_nc=3, nf=64)

        self.prompt_scale1 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale2 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale3 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale4 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale5 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale6 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale7 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale8 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale9 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale10 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale11 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale12 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale13 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale14 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale15 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scale16 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scalehr = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_scaleout = nn.Linear(prompt_nf, 3, bias=True)

        self.prompt_shift1 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift2 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift3 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift4 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift5 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift6 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift7 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift8 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift9 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift10 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift11 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift12 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift13 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift14 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift15 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shift16 = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shifthr = nn.Linear(prompt_nf, num_feat, bias=True)
        self.prompt_shiftout = nn.Linear(prompt_nf, 3, bias=True)


        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body3 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body4 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body5 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body6 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body7 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body8 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body9 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body10 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body11 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body12 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body13 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body14 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body15 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)
        self.body16 = make_layer(ResidualBlockNoBN, 1, num_feat=num_feat)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):

        prompt = x[1]
        prompt = self.F_ext_net(prompt)

        scale1 = self.prompt_scale1(prompt)
        shift1 = self.prompt_shift1(prompt)

        scale2 = self.prompt_scale2(prompt)
        shift2 = self.prompt_shift2(prompt)

        scale3 = self.prompt_scale3(prompt)
        shift3 = self.prompt_shift3(prompt)

        scale4 = self.prompt_scale4(prompt)
        shift4 = self.prompt_shift4(prompt)

        scale5 = self.prompt_scale5(prompt)
        shift5 = self.prompt_shift5(prompt)

        scale6 = self.prompt_scale6(prompt)
        shift6 = self.prompt_shift6(prompt)

        scale7 = self.prompt_scale7(prompt)
        shift7 = self.prompt_shift7(prompt)

        scale8 = self.prompt_scale8(prompt)
        shift8 = self.prompt_shift8(prompt)

        scale9 = self.prompt_scale9(prompt)
        shift9 = self.prompt_shift9(prompt)

        scale10 = self.prompt_scale10(prompt)
        shift10 = self.prompt_shift10(prompt)

        scale11 = self.prompt_scale11(prompt)
        shift11 = self.prompt_shift11(prompt)

        scale12 = self.prompt_scale12(prompt)
        shift12 = self.prompt_shift12(prompt)

        scale13 = self.prompt_scale13(prompt)
        shift13 = self.prompt_shift13(prompt)

        scale14 = self.prompt_scale14(prompt)
        shift14 = self.prompt_shift14(prompt)

        scale15 = self.prompt_scale15(prompt)
        shift15 = self.prompt_shift15(prompt)

        scale16 = self.prompt_scale16(prompt)
        shift16 = self.prompt_shift16(prompt)

        scalehr = self.prompt_scalehr(prompt)
        shifthr = self.prompt_shifthr(prompt)

        scaleout = self.prompt_scaleout(prompt)
        shiftout = self.prompt_shiftout(prompt)


        feat = self.lrelu(self.conv_first(x[0]))

        feat = self.body1(feat)
        feat = feat * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body2(feat)
        feat = feat * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body3(feat)
        feat = feat * scale3.view(-1, self.base_nf, 1, 1) + shift3.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body4(feat)
        feat = feat * scale4.view(-1, self.base_nf, 1, 1) + shift4.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body5(feat)
        feat = feat * scale5.view(-1, self.base_nf, 1, 1) + shift5.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body6(feat)
        feat = feat * scale6.view(-1, self.base_nf, 1, 1) + shift6.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body7(feat)
        feat = feat * scale7.view(-1, self.base_nf, 1, 1) + shift7.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body8(feat)
        feat = feat * scale8.view(-1, self.base_nf, 1, 1) + shift8.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body9(feat)
        feat = feat * scale9.view(-1, self.base_nf, 1, 1) + shift9.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body10(feat)
        feat = feat * scale10.view(-1, self.base_nf, 1, 1) + shift10.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body11(feat)
        feat = feat * scale11.view(-1, self.base_nf, 1, 1) + shift11.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body12(feat)
        feat = feat * scale12.view(-1, self.base_nf, 1, 1) + shift12.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body13(feat)
        feat = feat * scale13.view(-1, self.base_nf, 1, 1) + shift13.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body14(feat)
        feat = feat * scale14.view(-1, self.base_nf, 1, 1) + shift14.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body15(feat)
        feat = feat * scale15.view(-1, self.base_nf, 1, 1) + shift15.view(-1, self.base_nf, 1, 1) + feat

        feat = self.body16(feat)
        feat = feat * scale16.view(-1, self.base_nf, 1, 1) + shift16.view(-1, self.base_nf, 1, 1) + feat


        feat = self.lrelu(self.conv_hr(feat))
        feat = feat * scalehr.view(-1, self.base_nf, 1, 1) + shifthr.view(-1, self.base_nf, 1, 1) + feat

        feat = self.conv_last(feat)
        out = feat * scaleout.view(-1, 3, 1, 1) + shiftout.view(-1, 3, 1, 1) + feat
        base = F.interpolate(x[0], scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out



