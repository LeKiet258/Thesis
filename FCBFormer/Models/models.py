from functools import partial
import numpy as np

import torch
from torch import nn

from Models import pvt_v2
from timm.models.vision_transformer import _cfg


class RB(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=352,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h

'''Focus on 'what' '''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # squeeze spatial dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global AvgPool: đầu vào là 1 stack of 3 feature map, mỗi map 3x3 (input: 3x3x3) --GlobalAvgPool--> output là 1 stack of 3 scalar (3x1x1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False) 
        self.relu1 = nn.ReLU() 
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)        

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h

class SAM(nn.Module):
    def __init__(self, num_in=64, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        '''Params:
        - x: T1, aka SFA, có shape = (88,88,64)
        - edge (trong đồ thị): T2, aka CIM, có shape =  F1's shape = (88,88,64)
        '''
        import torch.nn.functional as F
        # edge = F.upsample(edge, (x.size()[-2], x.size()[-1])) # upsample edge T2 thành kích thước (h,w) của T1, tức edge mới có shape (88,88,64)

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1) # apply F(.) on T2 to produce T2' (88,88,1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1) # Q (n,16,h,w)
        x_proj = self.conv_proj(x) # K (88,88,16)
        x_mask = x_proj * edge # elem-wise(K,T2')

        # AdaptivePool(V)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1)) # matmul(K,V)
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1) # f = softmax(matmul(K,V))

        x_rproj_reshaped = x_proj_reshaped # f_copy = copy(f), do sau này có dùng lại f

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1)) # matmul(Q,f)
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state) # GCN(matmul(Q,f))

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped) # Y' = f @ GCN(matmul(Q,f))
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:]) # reshape thành (n,16,h,w)
        out = x + (self.conv_extend(x_state)) # T1 + Wz(Y')
        # print(f"SAM block, out.shape = {out.shape}") # expect: (h,w,c) = (88,88,64)

        return out    


class TB(nn.Module):
    def __init__(self):

        super().__init__()

        # backbone: sdụng pvt_v2_b3
        backbone = pvt_v2.PyramidVisionTransformerImpr(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1
        )

        checkpoint = torch.load("pvt_v2_b3.pth")
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()} # note: class PVTV2Impr đã bỏ đi head
        
        model_dict.update(state_dict)
        backbone.default_cfg = _cfg()
        backbone.load_state_dict(model_dict)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1] # remove -1 là gì ?????????

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        # LE block: F1-4 khi qua LE đều cho output shape (88,88,64) = F1's shape
        self.LE = nn.ModuleList([])
        for i in range(4): # 4 cục LE
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB(64, 64), nn.Upsample(size=88)
                )
            )

        # CIM
        self.ca = ChannelAttention(64) # F1 theo hình lun có chiều sâu = 64
        self.sa = SpatialAttention() 

        # SFA block
        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

        # SAM
        self.SAM = SAM()

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]

        for i, module in enumerate(self.backbone): # through 4 stage
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    # print(f"sub_module: {sub_module}")
                    x = sub_module(x, H, W)
            else: # i in [2,5,8]
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)
        # i cuối là nhiu? 
        # feature map x1->x4 đựng trong pyramid

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = [] # emph = emphasis
        # đi qua PLD+: LE + SFA
        # đi qua LE
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        # chỉnh CIM (pyramid_emph[0]) ở đây
        emph_0 = self.ca(pyramid_emph[0]) * pyramid_emph[0] # channel attention, hadarmart product
        cim_feature = self.sa(emph_0) * emph_0 # spatial attention, hadarmart product

        # đi qua SFA 
        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1): # sfa from top to bot
            l = torch.cat((pyramid_emph[i], l_i), dim=1) # F_32, F_321, F_3210
            l = self.SFA[i](l) # đi qua RB,RB
            l_i = l

        sam_feature = self.SAM(l, cim_feature) # SAM(SFA, CIM)

        return sam_feature


class FCBFormer(nn.Module):
    def __init__(self, size=352):

        super().__init__()

        self.TB = TB()

        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            RB(64 + 32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x):
        x1 = self.TB(x)
        x2 = self.FCB(x)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)

        return out

