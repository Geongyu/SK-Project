
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet





# For Classifications 

def efficientnet(number, num_classes) :
    return EfficientNet.from_pretrained('efficientnet-' + number, num_classes=num_classes)


# Coordconv Module

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
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


def make_coordconvlist(encodernumber, coordconv):
    """

    make_coordconvlist : Function to apply coordconv of each layer

    Args:
        encodernumber: No of all layer
        coordconv: The layer number to apply coordconv

    """

    coordconv_list = [False for i in range(encodernumber)]

    if coordconv != None:

        for i in coordconv:
            coordconv_list[i] = True

    assert len(coordconv_list) == encodernumber, 'not match coordconv_list'

    return coordconv_list


# Uent Module

class ConvBnRelu(nn.Module):
    """
    conv -> bn -> relu

    Args:
        coordconv: if True, coordconv -> bn -> relu

    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, momentum=0.1, coordconv=False,
                 radius=False):
        super(ConvBnRelu, self).__init__()

        if coordconv:
            self.conv = CoordConv(in_channels, out_channels, kernel_size=1,
                                  padding=0, stride=1, with_r=radius)
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

        x = self.transpose_conv(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)

        return x


# Squeeze and Excitation Module

class ChannelSELayer(nn.Module):
    """
    ChannelSELayer : squeezing spatially and exciting channel-wise

    Args:
        num_channels: No of input channels
        reduction_ratio: By how much should the num_channels should be reduced

    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
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
    SpatialSELayer : squeezing channel-wise and exciting spatially

    Args:
        num_channels: No of input channels

    """

    def __init__(self, num_channels):

        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):

        """
        Args:
            weights: weights for few shot learning
            input_tensor: X, shape = (batch_size, num_channels, H, W)

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
    """
    ChannelSpatialSELayer : concurrent spatial and channel squeeze & excitation

    Args:
        num_channels: No of input channels
        reduction_ratio: By how much should the num_channels should be reduced

    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)

        return output_tensor


######################################################################################################

# model

class Unet2D(nn.Module):
    """
    Args:
        momentum : batchNorm momentum defalt 0.1

    """

    def __init__(self, in_shape, padding=1, momentum=0.1):
        super(Unet2D, self).__init__()
        channels, heights, width = in_shape
        self.padding = padding

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum)
        self.down2 = StackEncoder(64, 128, padding, momentum=momentum)
        self.down3 = StackEncoder(128, 256, padding, momentum=momentum)
        self.down4 = StackEncoder(256, 512, padding, momentum=momentum)

        self.center1 = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

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

        return out


# Unet + scaled input

class Unet2D_multiinput(nn.Module):
    """
    Multi-scale input : 각 인코더 block에 scaled input 추가

    *Abraham and Khan, (2018), A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation*

    """

    def __init__(self, in_shape, padding=1, momentum=0.1):
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

        self.multiinput1 = nn.Sequential(nn.Upsample(size=(heights // 2, width // 2), mode='bilinear'),
                                         ConvBnRelu(64, 64, kernel_size=(3, 3), stride=1, padding=padding,
                                                    momentum=momentum))

        self.multiinput2 = nn.Sequential(nn.Upsample(size=(heights // 4, width // 4), mode='bilinear'),
                                         ConvBnRelu(128, 128, kernel_size=(3, 3), stride=1, padding=padding,
                                                    momentum=momentum))

        self.multiinput3 = nn.Sequential(nn.Upsample(size=(heights // 8, width // 8), mode='bilinear'),
                                         ConvBnRelu(256, 256, kernel_size=(3, 3), stride=1, padding=padding,
                                                    momentum=momentum))

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        multiinput1 = self.multiinput1(x)
        multiinput1 = torch.cat((x, multiinput1), 1)

        x, x_trace2 = self.down2(multiinput1)
        multiinput2 = self.multiinput2(x)
        multiinput2 = torch.cat((x, multiinput2), 1)

        x, x_trace3 = self.down3(multiinput2)
        multiinput3 = self.multiinput3(x)
        multiinput3 = torch.cat((x, multiinput3), 1)

        x, x_trace4 = self.down4(multiinput3)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        return out


# Unet + coordconv

class Unet2D_coordconv(nn.Module):
    """
    CoordConv: 위치 정보를 입력으로 반영

    Args:
        coordnumber: 해당 레이어의 coordconv 여부
        radius: coordconv를 적용시 radius 적용 여부


    *Liu et al. (2018), An intriguing failing of convolutional neural networks and the coordconv solution*

    """

    def __init__(self, in_shape, padding=1, momentum=0.1, coordnumber=None, radius=False):
        super(Unet2D_coordconv, self).__init__()
        channels, heights, width = in_shape

        if coordnumber:
            coordconv_list = make_coordconvlist(10, coordnumber)

        self.down1 = StackEncoder(channels, 64, padding, momentum=momentum, coordconv=coordconv_list[0], radius=radius)
        self.down2 = StackEncoder(64, 128, padding, momentum=momentum, coordconv=coordconv_list[1], radius=radius)
        self.down3 = StackEncoder(128, 256, padding, momentum=momentum, coordconv=coordconv_list[2], radius=radius)
        self.down4 = StackEncoder(256, 512, padding, momentum=momentum, coordconv=coordconv_list[3], radius=radius)

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum,
                       coordconv=coordconv_list[4], radius=radius),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum,
                       coordconv=coordconv_list[5], radius=radius)
        )

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum,
                                coordconv=coordconv_list[6], radius=radius)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum,
                                coordconv=coordconv_list[7], radius=radius)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum,
                                coordconv=coordconv_list[8], radius=radius)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum,
                                coordconv=coordconv_list[9], radius=radius)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

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


# Unet + squeeze_and_excitation

class Unet_sae(nn.Module):
    """
    scSE block
        - channel 정보와 spatial 정보를 반영한 block
        - unet의 마지막 encoder layer 뒤에 적용

    *Roy et al. (2018), Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks*

    """

    def __init__(self, in_shape, padding=1, momentum=0.1):
        super(Unet_sae, self).__init__()
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
        self.sae_layer = ChannelSpatialSELayer(1024)

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, padding=padding, momentum=momentum)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)
        x = self.sae_layer(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)

        return out


if __name__ == '__main__':
    from torchsummary import summary

    # net = Unet_sae(in_shape=(1, 512, 512), padding=1,momentum=0.1)
    #
    # net = Unet2D(in_shape=(1, 512, 512), padding=1, momentum=0.1)
    net = Unet_sae()
    summary(net.cuda(), ((1, 512, 512)))
    import ipdb;

    ipdb.set_trace()
