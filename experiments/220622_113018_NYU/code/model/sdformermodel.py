import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numbers
from einops import rearrange


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
class Attention(nn.Module):
    def __init__(self, window_sizes, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.split_chns = [dim, dim, dim]
        self.window_sizes = window_sizes

    def forward(self, x):
        x = self.qkv_dwconv(self.qkv(x))
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        for idx, x_ in enumerate(xs):
            b, c, h, w = x_.shape
            wsize = self.window_sizes[idx]
            q, k, v = x_.chunk(3, dim=1)
            q = rearrange(q, 'b (head c) (h dh) (w dw) ->  (b) (h w) (head c) (dh dw)', head=self.num_heads, dh=wsize[0], dw=wsize[1])
            k = rearrange(k, 'b (head c) (h dh) (w dw) ->  (b) (h w) (head c) (dh dw)', head=self.num_heads, dh=wsize[0], dw=wsize[1])
            v = rearrange(v, 'b (head c) (h dh) (w dw) ->  (b) (h w) (head c) (dh dw)', head=self.num_heads, dh=wsize[0], dw=wsize[1])

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

            out = (attn @ v)
            out = rearrange(out, '(b) (h w) (head c) (dh dw) -> b (head c) (h dh) (w dw)', head=self.num_heads, h=h // wsize[0], w=w // wsize[1],
                            dh=wsize[0], dw=wsize[1])
            ys.append(out)
        y = torch.cat(ys, dim=1)
        y = self.project_out(y)
        return y


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, window_sizes, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(window_sizes, dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################

class SDFORMERModel(nn.Module):
    def __init__(self, args):
        super(SDFORMERModel, self).__init__()
        self.args = args
        self.inp_channels = args.inp_channels
        self.out_channels = args.out_channels
        self.dim = args.dim
        self.heads = args.heads
        self.num_blocks = args.num_blocks
        self.num_refinement_blocks = args.num_refinement_blocks
        self.ffn_expansion_factor = args.ffn_expansion_factor
        self.bias = args.bias
        self.LayerNorm_type = args.LayerNorm_type

        self.window_sizes1 = args.window_sizes1
        self.window_sizes2 = args.window_sizes2
        self.window_sizes3 = args.window_sizes3
        self.window_sizes4 = args.window_sizes4

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(self.window_sizes1,
                               dim=self.dim,
                               num_heads=self.heads[0],
                               ffn_expansion_factor=self.ffn_expansion_factor,
                               bias=self.bias,
                               LayerNorm_type=self.LayerNorm_type)
              for i in range(self.num_blocks[0])])

        self.down1_2 = Downsample(self.dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(self.window_sizes2,
                             dim=int(self.dim * 2 ** 1),
                             num_heads=self.heads[1],
                             ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=self.bias,
                             LayerNorm_type=self.LayerNorm_type)
            for i in range(self.num_blocks[1])])

        self.down2_3 = Downsample(int(self.dim * 2 ** 1))

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(self.window_sizes3,
                             dim=int(self.dim * 2 ** 2),
                             num_heads=self.heads[2],
                             ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=self.bias,
                             LayerNorm_type=self.LayerNorm_type)
            for i in range(self.num_blocks[2])])

        self.down3_4 = Downsample(int(self.dim * 2 ** 2))

        self.latent = nn.Sequential(*[
            TransformerBlock(self.window_sizes4,
                             dim=int(self.dim * 2 ** 3),
                             num_heads=self.heads[3],
                             ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=self.bias,
                             LayerNorm_type=self.LayerNorm_type)
            for i in range(self.num_blocks[3])])

        self.up4_3 = Upsample(int(self.dim * 2 ** 3))

        self.reduce_chan_level3 = nn.Conv2d(int(self.dim * 2 ** 3), int(self.dim * 2 ** 2), kernel_size=1, bias=self.bias)

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(self.window_sizes3,
                             dim=int(self.dim * 2 ** 2),
                             num_heads=self.heads[2],
                             ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=self.bias,
                             LayerNorm_type=self.LayerNorm_type)
            for i in range(self.num_blocks[2])])

        self.up3_2 = Upsample(int(self.dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(self.dim * 2 ** 2), int(self.dim * 2 ** 1), kernel_size=1, bias=self.bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(self.window_sizes2,
                             dim=int(self.dim * 2 ** 1),
                             num_heads=self.heads[1],
                             ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=self.bias,
                             LayerNorm_type=self.LayerNorm_type)
            for i in range(self.num_blocks[1])])

        self.up2_1 = Upsample(int(self.dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(self.window_sizes1,
                             dim=int(self.dim * 2 ** 1),
                             num_heads=self.heads[0],
                             ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=self.bias,
                             LayerNorm_type=self.LayerNorm_type)
            for i in range(self.num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(self.window_sizes1,
                             dim=int(self.dim * 3),
                             num_heads=self.heads[0],
                             ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=self.bias,
                             LayerNorm_type=self.LayerNorm_type)
            for i in range(self.num_refinement_blocks)])

        self.output = nn.Conv2d(int(self.dim * 3), self.out_channels, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.conv1_rgb = conv_bn_relu(3, 18, kernel=3, stride=1, padding=1, bn=False)

        self.conv1_dep = conv_bn_relu(1, 6, kernel=3, stride=1, padding=1, bn=False)

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]

    def _beforeDown(self, a):
        _, _, Hd, Wd = a.shape
        if Hd % 2 != 0:
            a = torch.cat((a, torch.unsqueeze(a[:, :, -1, :], dim=2)), dim=2)

        if Wd % 2 != 0:
            a = torch.cat((a, torch.unsqueeze(a[:, :, :, -1], dim=3)), dim=3)

        return a

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']

        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)

        inp_enc_level1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(self._beforeDown(out_enc_level1))
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(self._beforeDown(out_enc_level2))
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(self._beforeDown(out_enc_level3))
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self._concat(inp_dec_level3, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self._concat(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self._concat(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self._concat(out_dec_level1, inp_enc_level1)
        r1 = self.refinement(out_dec_level1)

        rx = self.output(r1)
        y = torch.clamp(rx, min=0)
        output = {'pred': y}

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_sizes1',
                        default=[[12, 16], [6, 8], [4, 4]],
                        help='window_sizes')
    parser.add_argument('--window_sizes2',
                        default=[[6, 19], [19, 8], [6, 4]],
                        help='window_sizes')
    parser.add_argument('--window_sizes3',
                        default=[[3, 19], [19, 4], [3, 4]],
                        help='window_sizes')
    parser.add_argument('--window_sizes4',
                        default=[[29, 38], [29, 19], [29, 2]],
                        help='window_sizes')
    parser.add_argument('--inp_channels',
                        default=4,
                        help='inp_channels')
    parser.add_argument('--out_channels',
                        default=1,
                        help='out_channels')
    parser.add_argument('--dim',
                        default=24,
                        help='dim')
    parser.add_argument('--num_blocks',
                        default=[2, 4, 6, 8],
                        help='num_blocks')
    parser.add_argument('--heads',
                        default=[1, 2, 4, 8],
                        help='heads')
    parser.add_argument('--num_refinement_blocks',
                        default=2,
                        help='num_refinement_blocks')
    parser.add_argument('--ffn_expansion_factor',
                        default=2.66,
                        help='ffn_expansion_factor')
    parser.add_argument('--bias',
                        default=False,
                        help='bias')
    parser.add_argument('--LayerNorm_type',
                        default='WithBias',
                        help='LayerNorm_type')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SDFORMERModel(args).to(device)
    print(model)
    rgb = torch.randn((1, 3, 228, 304)).to(device)
    dep = torch.randn((1, 1, 228, 304)).to(device)
    sample = {'rgb': rgb, 'dep': dep}
    out = model(sample)
    print('------------finish-----------------------------')
