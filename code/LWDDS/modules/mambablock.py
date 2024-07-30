from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().reshape(B, C, nf, H, W)
        x = self.dwconv(x)
        x = x.contiguous().flatten(2).transpose(1, 2)

        return x



class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, nf, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
        
        



class MambaLayerglobal(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU, reverse=True):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
                # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.reverse = reverse
        self.apply(self._init_weights)
       


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        

        x = x.permute(0, 2, 1, 3, 4)
        B, C, nf, H, W = x.shape

        assert C == self.dim        

        n_tokens = x.shape[2:].numel()

        img_dims = x.shape[2:]

        
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        # Bi-Mamba layer
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))

 
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        
        out = out.permute(0, 2, 1, 3, 4)
      
        return out





class MambaLayerlocal(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU, reverse=True):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
                # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.reverse = reverse
        self.apply(self._init_weights)
       


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, hilbert_curve):

        x = x.permute(0, 2, 1, 3, 4)
        B, C, nf, H, W = x.shape

        if self.reverse:
            x = x.permute(0,1,3,4,2)
        assert C == self.dim        

        img_dims = x.shape[2:]
        x_hw = x.flatten(2).contiguous()

        x_hil = x_hw.index_select(dim=-1, index=hilbert_curve)
        x_flat = x_hil.transpose(-1, -2)

        # Bi-Mamba layer
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba_out = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        outmamba = x_mamba_out.transpose(-1, -2)


        sum_out = torch.zeros_like(outmamba)
        hilbert_curve_re = repeat(hilbert_curve, 'hw -> b c hw', b=outmamba.shape[0], c=outmamba.shape[1])
        assert outmamba.shape == hilbert_curve_re.shape

        sum_out.scatter_add_(dim=-1, index=hilbert_curve_re, src=outmamba)
        sum_out = sum_out.reshape(B, C, *img_dims).contiguous()

        if self.reverse:
            out = sum_out.permute(0,1,4,2,3)
        
        out = out.permute(0, 2, 1, 3, 4)
      
        return out