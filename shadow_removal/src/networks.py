import torch
import torch.nn as nn
# from kpn.network import KernelConv
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class BranchUp(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8):
        super(BranchUp, self).__init__()

        self.kernel_size = config.kernel_size

        self.encoder0 = nn.Sequential(
            # nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=3),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks-1):
            block = ResnetBlock(256)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=3)
        )

    def get_kernel_feat(self, feat):
        kernels = self.kernels_feat(feat)
        kernels = kernels.unsqueeze(dim=0)
        kernels = F.interpolate(input=kernels, size=(256 * 9, feat.shape[-1], feat.shape[-2]), mode='nearest')
        kernels = kernels.squeeze(dim=0)

        return kernels

    def get_kernel_img(self, feat):
        kernels = self.kernel_img(feat)
        return kernels


class BranchDown(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8):
        super(BaseNetwork, self).__init__()

        self.encoder0 = nn.Sequential(
            # nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=3),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=3)
        )


class InpaintGenerator(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        config.kernel_size = [3]

        self.branch_up = BranchUp(config=config, residual_blocks=residual_blocks)
        self.branch_down = BranchDown(config=config, residual_blocks=residual_blocks)

        # self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kernel_64 = nn.Conv2d(in_channels=512, out_channels=2*256 * 1, kernel_size=3, stride=1, padding=1)
        self.fuse_64 = nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fuse_256 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.kernel_256 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)


        if init_weights:
            self.init_weights()

    def forward(self, input_up, input_down):

        up_1 = self.branch_up.encoder0(input_up)
        up_2 = self.branch_up.encoder1(up_1)
        up_3 = self.branch_up.encoder2(up_2)

        # -------------------------------------------------------------
        down_1 = self.branch_down.encoder0(input_down)
        down_2 = self.branch_down.encoder1(down_1)
        down_3 = self.branch_down.encoder2(down_2)

        # ---------------------------------
        feat_cat = torch.cat([up_3, down_3], dim=1)
        feat_k = self.kernel_64(feat_cat)
        feat_k = torch.sigmoid(feat_k)
        feat_w = feat_k * feat_cat
        feat_fuse = self.fuse_64(feat_w)
        # --------------------------------

        down_4 = self.branch_down.middle(feat_fuse)
        down_5 = self.branch_down.decoder_0(down_4)
        down_6 = self.branch_down.decoder_1(down_5)
        down_7 = self.branch_down.decoder_2(down_6)

        img_fuse = (torch.tanh(down_7) + 1) / 2

        return img_fuse


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
