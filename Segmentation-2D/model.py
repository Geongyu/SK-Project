import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


#########################################################################################################

# For Classifications 

def efficientnet(number, num_classes) :
    return EfficientNet.from_pretrained('efficientnet-' + number, num_classes=num_classes)


# coordconv =========================================================================================

class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        # print('before coordconv ',x.size())
        ret = self.addcoords(x)
        # print('after coordconv ', ret.size())
        ret = self.conv(ret)
        return ret


class CoordConv_block(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('before coordconv ',x.size())
        ret = self.addcoords(x)
        # print('after coordconv ', ret.size())
        ret = self.conv(ret)
        ret = self.sigmoid(ret)

        batch_size, channel, a, b = x.size()
        # spatial excitation
        output_tensor = torch.mul(x, ret.view(batch_size, 1, a, b))

        return output_tensor


def TF_coordconv(encodernumber, coordconv):
    TF_coordconv_list = []
    if coordconv == None:
        TF_coordconv_list = [False for i in range(encodernumber)]
    else:
        for i in range(0, encodernumber):
            if i in coordconv:
                TF_coordconv_list.append(True)
            else:
                TF_coordconv_list.append(False)

    assert len(TF_coordconv_list) == encodernumber, 'not match coordconv_list'

    return TF_coordconv_list


# coordconv =========================================================================================


# 2d model

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, momentum=0.1, coordconv=False,
                 radius=False):
        super(ConvBnRelu, self).__init__()

        if coordconv:
            # 1x1 conv
            self.conv = CoordConv(in_channels, out_channels, kernel_size=1,
                                  padding=0, stride=1, with_r=radius)
            # self.conv = CoordConv(in_channels, out_channels, kernel_size=kernel_size,
            #                     padding=1, stride=1, with_r=radius)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=padding, stride=stride)

        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder, self).__init__()

        # self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        # x = self.upSample(x)
        x = self.transpose_conv(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x


class StackEncoder_skip(nn.Module):
    def __init__(self, in_channels, out_channels, padding, calculate, momentum=0.1, coordconv=False, radius=False):
        super(StackEncoder_skip, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # if cal = concat
        self.down_useconcat = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1)
        self.down_Bn = nn.BatchNorm2d(out_channels)
        self.calculate = calculate

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        identity = self.down(x)
        identity = self.down_Bn(identity)

        x = self.convr1(x)
        x = self.convr2(x)

        x = calculate_mode(self.calculate, x, identity)
        if self.calculate == 'concat':
            x = self.down_useconcat(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder_skip(nn.Module):
    def __init__(self, in_channels, out_channels, padding, calculate, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder_skip, self).__init__()

        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        # if cal == concat
        self.down_useconcat = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1)
        self.down_Bn = nn.BatchNorm2d(out_channels)
        self.calculate = calculate

        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        # x = self.upSample(x)

        x = self.transpose_conv(x)
        identity = self.down(down_tensor)
        identity = self.down_Bn(identity)

        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)

        x = calculate_mode(self.calculate, x, identity)
        if self.calculate == 'concat':
            x = self.down_useconcat(x)
        return x


# use attnUnet
class StackDecoder_attn(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder_attn, self).__init__()

        # self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

        self.attn_block = Attention_block(F_g=out_channels, F_l=out_channels, F_int=out_channels)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        # x = self.upSample(x)
        x = self.transpose_conv(x)
        print(x.shape)
        print(down_tensor.shape)
        attn = self.attn_block(x, down_tensor)
        print('here')
        x = self._crop_concat(x, attn)
        x = self.convr1(x)
        x = self.convr2(x)
        return x


class res_calculate():
    def sum(x, identity):
        x = x + identity
        return x

    def multiple(x, identity):
        x = x * identity
        return x

    def concat(x, identity):
        x = torch.cat((x, identity), 1)
        return x


def calculate_mode(mode, x, identy):
    if mode == 'sum':
        calculate = res_calculate.sum(x, identy)
    elif mode == 'multiple':
        calculate = res_calculate.multiple(x, identy)
    elif mode == 'concat':
        calculate = res_calculate.concat(x, identy)

    return calculate


# attention model

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        print('#' * 50)
        print(g.shape)
        g1 = self.W_g(g)
        print('@' * 50)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# Squeeze and Excitation Module


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor


class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, coordconv=True, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
        # self.coord = CoordConv_block(num_channels,num_channels)

        self.coord_mode = coordconv

        # def __init__(self, in_channels, out_channels, with_r=False, **kwargs):

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        """
        if self.coord_mode :
            output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor) + self.coord(input_tensor)
        else:
            output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)
        """

        output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)

        return output_tensor


class ChannelSpatialSELayer_coord(nn.Module):

    def __init__(self, num_channels, coordconv=True, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
        self.coord = CoordConv_block(num_channels, num_channels)

        self.coord_mode = coordconv

        # def __init__(self, in_channels, out_channels, with_r=False, **kwargs):

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))

        if self.coord_mode:
            output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor) + self.coord(input_tensor)
        else:
            output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)

        return output_tensor


######################################################################################################
# model
## 필터수 절반으로 줄이고

class Unet2D(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding, momentum):
        super(Unet2D, self).__init__()
        channels, heights, width = in_shape
        self.padding = padding

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum)
        self.down2 = StackEncoder(64, 128, padding, momentum=momentum)
        self.down3 = StackEncoder(128, 256, padding, momentum=momentum)
        self.down4 = StackEncoder(256, 512, padding, momentum=momentum)

        self.center1 = ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        self.center2 = ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x_bottom_block = self.center1(x)
        x = self.center2(x_bottom_block)
        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        # need size down output
        if self.padding == 0:
            out = self.output_up_seg_map(out)

        return out, x_bottom_block


class Unet2D_multiinput(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding, momentum):
        super(Unet2D_multiinput, self).__init__()
        channels, heights, width = in_shape
        self.padding = padding

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum)
        self.down2 = StackEncoder(64 * 2, 128, padding, momentum=momentum)
        self.down3 = StackEncoder(128 * 2, 256, padding, momentum=momentum)
        self.down4 = StackEncoder(256 * 2, 512, padding, momentum=momentum)

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

        self.multiinput1 = nn.Upsample(size=(heights // 2, width // 2), mode='bilinear')
        self.multiinput2 = nn.Upsample(size=(heights // 4, width // 4), mode='bilinear')
        self.multiinput3 = nn.Upsample(size=(heights // 8, width // 8), mode='bilinear')

        self.multiinput1_3x3 = ConvBnRelu(64, 64, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        self.multiinput2_3x3 = ConvBnRelu(128, 128, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        self.multiinput3_3x3 = ConvBnRelu(256, 256, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)

    def forward(self, x):
        input = x
        x, x_trace1 = self.down1(x)
        multiinput1 = self.multiinput1(x)
        multiinput1 = self.multiinput1_3x3(multiinput1)
        multiinput1 = torch.cat((x, multiinput1), 1)

        x, x_trace2 = self.down2(multiinput1)
        multiinput2 = self.multiinput2(x)
        multiinput2 = self.multiinput2_3x3(multiinput2)
        multiinput2 = torch.cat((x, multiinput2), 1)

        x, x_trace3 = self.down3(multiinput2)
        multiinput3 = self.multiinput3(x)
        multiinput3 = self.multiinput3_3x3(multiinput3)
        multiinput3 = torch.cat((x, multiinput3), 1)

        x, x_trace4 = self.down4(multiinput3)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        # need size down output
        if self.padding == 0:
            out = self.output_up_seg_map(out)

        return out


class Unetcoordconv(nn.Module):
    """
    coordconv unet : padding=1,momentum=0.1,coordconv = True
    """

    def __init__(self, in_shape, padding, momentum, coordnumber=None, radius=False):
        super(Unetcoordconv, self).__init__()
        channels, heights, width = in_shape
        encodernumber = 10
        padding = 1

        if coordnumber:
            TF_coordconv_list = TF_coordconv(encodernumber, coordnumber)
        else:
            TF_coordconv_list = TF_coordconv(encodernumber, coordnumber)

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum, coordconv=TF_coordconv_list[0],
                                  radius=radius)
        self.down2 = StackEncoder(64, 128, padding, momentum=momentum, coordconv=TF_coordconv_list[1], radius=radius)
        self.down3 = StackEncoder(128, 256, padding, momentum=momentum, coordconv=TF_coordconv_list[2], radius=radius)
        self.down4 = StackEncoder(256, 512, padding, momentum=momentum, coordconv=TF_coordconv_list[3], radius=radius)

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum,
                       coordconv=TF_coordconv_list[4], radius=radius),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum,
                       coordconv=TF_coordconv_list[5], radius=radius)
        )

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum,
                                coordconv=TF_coordconv_list[6], radius=radius)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum,
                                coordconv=TF_coordconv_list[7], radius=radius)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum,
                                coordconv=TF_coordconv_list[8], radius=radius)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum,
                                coordconv=TF_coordconv_list[9], radius=radius)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

    def forward(self, x):

        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        return out


class UnetSkipConnection(nn.Module):
    """
    coordconv unet : padding=1,momentum=0.1,coordconv = True
    """

    def __init__(self, in_shape, padding, momentum, calculate, coordnumber=None, radius=False):
        super(UnetSkipConnection, self).__init__()
        channels, heights, width = in_shape
        encodernumber = 10
        self.padding = padding

        if coordnumber:
            TF_coordconv_list = TF_coordconv(encodernumber, coordnumber)
        else:
            TF_coordconv_list = TF_coordconv(encodernumber, coordnumber)

        self.down1 = StackEncoder_skip(channels, 64, padding, calculate=calculate, momentum=momentum,
                                       coordconv=TF_coordconv_list[0], radius=radius)
        self.down2 = StackEncoder_skip(64, 128, padding, calculate=calculate, momentum=momentum,
                                       coordconv=TF_coordconv_list[1], radius=radius)
        self.down3 = StackEncoder_skip(128, 256, padding, calculate=calculate, momentum=momentum,
                                       coordconv=TF_coordconv_list[2], radius=radius)
        self.down4 = StackEncoder_skip(256, 512, padding, calculate=calculate, momentum=momentum,
                                       coordconv=TF_coordconv_list[3], radius=radius)

        self.down = nn.Conv2d(512, 1024, kernel_size=1, stride=1)
        # if cal == concat
        self.down_useconcat = nn.Conv2d(1024 * 2, 1024, kernel_size=1, stride=1)
        self.down_Bn = nn.BatchNorm2d(1024)
        self.calculate = calculate

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum,
                       coordconv=TF_coordconv_list[4], radius=radius),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum,
                       coordconv=TF_coordconv_list[5], radius=radius)
        )

        self.up1 = StackDecoder_skip(in_channels=1024, out_channels=512, padding=padding, calculate=calculate,
                                     momentum=momentum, coordconv=TF_coordconv_list[6], radius=radius)
        self.up2 = StackDecoder_skip(in_channels=512, out_channels=256, padding=padding, calculate=calculate,
                                     momentum=momentum, coordconv=TF_coordconv_list[7], radius=radius)
        self.up3 = StackDecoder_skip(in_channels=256, out_channels=128, padding=padding, calculate=calculate,
                                     momentum=momentum, coordconv=TF_coordconv_list[8], radius=radius)
        self.up4 = StackDecoder_skip(in_channels=128, out_channels=64, padding=padding, calculate=calculate,
                                     momentum=momentum, coordconv=TF_coordconv_list[9], radius=radius)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

    def forward(self, x):

        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        identity = self.down(x)
        identity = self.down_Bn(identity)

        x = self.center(x)
        x = calculate_mode(self.calculate, x, identity)

        if self.calculate == 'concat':
            x = self.down_useconcat(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        # need size down output
        if self.padding == 0:
            out = self.output_up_seg_map(out)

        return out


# attention unet
class Attn_Unet2D(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding, momentum):
        super(Attn_Unet2D, self).__init__()
        channels, heights, width = in_shape
        self.padding = padding

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum)
        self.down2 = StackEncoder(64, 128, padding, momentum=momentum)
        self.down3 = StackEncoder(128, 256, padding, momentum=momentum)
        self.down4 = StackEncoder(256, 512, padding, momentum=momentum)

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )
        # self.att1 =Attention_block(F_g=1024,F_l=1024,F_int=512)
        self.up1 = StackDecoder_attn(in_channels=1024, out_channels=512, padding=padding, momentum=momentum)
        # self.att2 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.up2 = StackDecoder_attn(in_channels=512, out_channels=256, padding=padding, momentum=momentum)
        # self.att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.up3 = StackDecoder_attn(in_channels=256, out_channels=128, padding=padding, momentum=momentum)
        # self.att4 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.up4 = StackDecoder_attn(in_channels=128, out_channels=64, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)
        print(x.shape)
        print(x_trace4.shape)
        # att1 =self.att1(x, x_trace4)
        x = self.up1(x, x_trace4)
        # att2 = self.att2(x, x_trace3)
        x = self.up2(x, x_trace3)
        # att3=self.att3(x, x_trace2)
        x = self.up3(x, x_trace2)
        # att4 = self.att4(x, x_trace1)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        # need size down output
        if self.padding == 0:
            out = self.output_up_seg_map(out)

        return out


# Unet + squeeze_and_excitation
class Unet_sae(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding, momentum, coordconv=False):
        super(Unet_sae, self).__init__()
        channels, heights, width = in_shape
        self.padding = padding

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum)
        self.sae_e1 = ChannelSpatialSELayer(64, coordconv)
        self.down2 = StackEncoder(64, 128, padding, momentum=momentum)
        self.sae_e2 = ChannelSpatialSELayer(128, coordconv)
        self.down3 = StackEncoder(128, 256, padding, momentum=momentum)
        self.sae_e3 = ChannelSpatialSELayer(256, coordconv)
        self.down4 = StackEncoder(256, 512, padding, momentum=momentum)
        self.sae_e4 = ChannelSpatialSELayer(512, coordconv)
        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )
        self.sae_c = ChannelSpatialSELayer(1024, coordconv)
        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum)
        self.sae_d1 = ChannelSpatialSELayer(512, coordconv)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum)
        self.sae_d2 = ChannelSpatialSELayer(256, coordconv)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum)
        self.sae_d3 = ChannelSpatialSELayer(128, coordconv)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum)
        self.sae_d4 = ChannelSpatialSELayer(64, coordconv)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x = self.sae_e1(x)
        x, x_trace2 = self.down2(x)
        x = self.sae_e2(x)
        x, x_trace3 = self.down3(x)
        x = self.sae_e3(x)
        x, x_trace4 = self.down4(x)
        x = self.sae_e4(x)

        x = self.center(x)
        x = self.sae_c(x)

        x = self.up1(x, x_trace4)
        x = self.sae_d1(x)
        x = self.up2(x, x_trace3)
        x = self.sae_d2(x)
        x = self.up3(x, x_trace2)
        x = self.sae_d3(x)
        x = self.up4(x, x_trace1)
        x = self.sae_d4(x)

        out = self.output_seg_map(x)

        # need size down output
        if self.padding == 0:
            out = self.output_up_seg_map(out)

        return out


class Unet_sae_p3(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding, momentum, coordconv=False):
        super(Unet_sae_p3, self).__init__()
        channels, heights, width = in_shape
        self.padding = padding

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum)
        # self.sae_e1 = ChannelSpatialSELayer(64,coordconv)
        self.down2 = StackEncoder(64, 128, padding, momentum=momentum)
        # self.sae_e2 = ChannelSpatialSELayer(128,coordconv)
        self.down3 = StackEncoder(128, 256, padding, momentum=momentum)
        # self.sae_e3 = ChannelSpatialSELayer(256,coordconv)
        self.down4 = StackEncoder(256, 512, padding, momentum=momentum)
        # self.sae_e4 = ChannelSpatialSELayer(512,coordconv)
        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )
        self.sae_c = ChannelSpatialSELayer(1024, coordconv)
        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum)
        # self.sae_d1 = ChannelSpatialSELayer(512,coordconv)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum)
        # self.sae_d2 = ChannelSpatialSELayer(256,coordconv)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum)
        # self.sae_d3 = ChannelSpatialSELayer(128,coordconv)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum)
        # self.sae_d4 = ChannelSpatialSELayer(64,coordconv)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(heights, width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        # x= self.sae_e1(x)
        x, x_trace2 = self.down2(x)
        # x = self.sae_e2(x)
        x, x_trace3 = self.down3(x)
        # x = self.sae_e3(x)
        x, x_trace4 = self.down4(x)
        # x = self.sae_e4(x)

        x = self.center(x)
        x = self.sae_c(x)

        x = self.up1(x, x_trace4)
        # x = self.sae_d1(x)
        x = self.up2(x, x_trace3)
        # x = self.sae_d2(x)
        x = self.up3(x, x_trace2)
        # x = self.sae_d3(x)
        x = self.up4(x, x_trace1)
        # x = self.sae_d4(x)

        out = self.output_seg_map(x)

        # need size down output
        if self.padding == 0:
            out = self.output_up_seg_map(out)

        return out


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, 3, padding=1),  # b, 16, 512, 512
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4),  # b, 16, 256, 256
            nn.Conv2d(256, 1024, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4),  # b, 8, 128, 128
            # nn.Conv2d(8, 4, 3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, stride=2)  # b, 8, 64, 64
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 1, 4, stride=4),
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 1, 2, stride=2),

        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder, encoder


class autoencoder_plus_input(nn.Module):
    def __init__(self):
        super(autoencoder_plus_input, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 256, 3, padding=1),  # b, 16, 512, 512
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4),  # b, 16, 256, 256
            nn.Conv2d(256, 1024, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4),  # b, 8, 128, 128
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 2, 4, stride=4),
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 1, 2, stride=2),

        )

    def forward(self, data):
        target, input = data
        x = torch.cat([target, input], dim=1)

        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        x = decoder[:, 0, ...]

        x = x.unsqueeze(1)

        return x, encoder


################################################################################################################################################################
# deeplabv3+

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None, momentum=0.01):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False, momentum=0.01):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])  # 64， 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])  # 128 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])  # 256 23
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1, momentum=0.01):
        """
        block class: 未初始化的bottleneck class
        planes:输出层数
        blocks:block个数
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1, momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, momentum=0.1):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding,
                                            dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=momentum)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True, momentum=0.1):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes       : {}".format(n_classes))
            print("Output stride           : {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
            print("Input shape             : {}".format("batchsize, 1, 512, 512"))
            print("Output shape            : {}".format("batchsize, 1, 512, 512"))

        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)
        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256, momentum=momentum),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256, momentum=momentum)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48, momentum=momentum)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256, momentum=momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256, momentum=momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):  # input 1, 3, 512, 512
        x, low_level_features = self.resnet_features(
            input)  # final_x:[1, 2048, 32, 32]  low_level_features:[1,256, 128, 128]
        x1 = self.aspp1(x)  # [1, 256, 32, 32]
        x2 = self.aspp2(x)  # [1, 256, 32, 32]
        x3 = self.aspp3(x)  # [1, 256, 32, 32]
        x4 = self.aspp4(x)  # [1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # [1, 256, 1, 1]
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2] / 4)), int(math.ceil(input.size()[-1] / 4))),
                          mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == '__main__':
    from torchsummary import summary

    # net = Unet_sae(in_shape=(1, 512, 512), padding=1,momentum=0.1)
    #
    # net = Unet2D(in_shape=(1, 512, 512), padding=1, momentum=0.1)
    net = autoencoder()
    summary(net.cuda(), ((1, 512, 512)))
    import ipdb;

    ipdb.set_trace()


