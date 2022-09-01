
class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, stride=1, padding=1, bias=False,groups=1, reduction=1):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * self.kernel_size * self.kernel_size, 1,bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        # self.bias = self.bias.repeat(b)
        weight = self.conv2(self.relu (self.bn(self.conv1(self.pool(x)))))
        # print('weight',weight.shape)
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        # print('weight',weight.shape)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias, stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x


def split_layer(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, stride, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        self.split_out_channels = split_layer(inplanes, scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(self.split_out_channels[i],self.split_out_channels[i], kernel_size = 2*i+3, stride=stride, padding=i+1, bias=False,groups=self.split_out_channels[i]))
            bns.append(nn.BatchNorm2d(self.split_out_channels[i]))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = nn.ReLU(inplace=True)
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        spx = torch.split(x, self.split_out_channels, dim=1)
        spk = []
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          #### final add
          if i == 0:
              out=sp
          else:
              out = torch.cat((out,sp),1)

        #   spk.append(sp)
        # out = torch.cat([s for s in (spk)], 1)

        return out
