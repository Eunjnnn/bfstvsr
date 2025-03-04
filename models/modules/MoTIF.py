'''
The code is modified from the implementation of Zooming Slow-Mo:
https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/codes/models/modules/Sakuya_arch.py
'''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.convlstm import ConvLSTM, ConvLSTMCell
try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
from models.modules.SIREN import Siren
from models.modules.warplayer import warpgrid
from models.core.raft import RAFT
import argparse
from models.softsplat_cp import Softsplat
from models.softsplat_max_cp import Softsplat_Max
from models.softsplat_count_cp import Softsplat_Count
from torch.nn.functional import interpolate, grid_sample
from einops import repeat


class TMB(nn.Module):

    def __init__(self):
        super(TMB, self).__init__()
        self.t_process = nn.Sequential(*[
            nn.Conv2d(1, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        ])
        self.f_process = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        ])

    def forward(self, x, t):
        feature = self.f_process(x)
        modulation_vector = self.t_process(t)
        output = feature * modulation_vector
        return output


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8, use_time=True):
        super(PCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if use_time == True:
            self.TMB_A_l1 = TMB()
            self.TMB_B_l1 = TMB()
            self.TMB_A_l2 = TMB()
            self.TMB_B_l2 = TMB()
            self.TMB_A_l3 = TMB()
            self.TMB_B_l3 = TMB()

    def forward(self, fea1, fea2, t=None, t_back=None):
        '''align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        '''
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(
            self.L3_offset_conv2_1(L3_offset)) if t is None else self.lrelu(self.L3_offset_conv2_1(L3_offset)) + self.TMB_A_l3(L3_offset, t)
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(
            self.L2_offset_conv3_1(L2_offset)) if t is None else self.lrelu(self.L2_offset_conv3_1(L2_offset)) + self.TMB_A_l2(L2_offset, t)
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(
            self.L1_offset_conv3_1(L1_offset)) if t is None else self.lrelu(self.L1_offset_conv3_1(L1_offset)) + self.TMB_A_l1(L1_offset, t)
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset)) if t_back is None else self.lrelu(
            self.L3_offset_conv2_2(L3_offset)) + self.TMB_B_l3(L3_offset, t_back)
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset)) if t_back is None else self.lrelu(
            self.L2_offset_conv3_2(L2_offset)) + self.TMB_B_l2(L2_offset, t_back)
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset)) if t_back is None else self.lrelu(
            self.L1_offset_conv3_2(L1_offset)) + self.TMB_B_l1(L1_offset, t_back)
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        y = torch.cat(y, dim=1)
        return y


class Easy_PCD(nn.Module):

    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        # input: extracted features
        # feature size: f1 = f2 = [B, N, C, H, W]
        # print(f1.size())
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea1 = [L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :, :].clone(), L3_fea[:, 0, :, :, :].clone()]
        fea2 = [L1_fea[:, 1, :, :, :].clone(), L2_fea[:, 1, :, :, :].clone(), L3_fea[:, 1, :, :, :].clone()]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea)  # [B, N, C, H, W]
        return fusion_fea


class DeformableConvLSTM(ConvLSTM):

    def __init__(self,
                 input_size,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 num_layers,
                 front_RBs,
                 groups,
                 batch_first=False,
                 bias=True,
                 return_all_layers=False):
        ConvLSTM.__init__(self,
                          input_size,
                          input_dim,
                          hidden_dim,
                          kernel_size,
                          num_layers,
                          batch_first=batch_first,
                          bias=bias,
                          return_all_layers=return_all_layers)
        #### extract features (for each frame)
        nf = input_dim

        self.pcd_h = Easy_PCD(nf=nf, groups=groups)
        self.pcd_c = Easy_PCD(nf=nf, groups=groups)

        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(input_size=(self.height, self.width),
                             input_dim=cur_input_dim,
                             hidden_dim=self.hidden_dim[i],
                             kernel_size=self.kernel_size[i],
                             bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input_tensor, hidden_state=None):
        '''
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None.

        Returns
        -------
        last_state_list, layer_output
        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3), input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), tensor_size=tensor_size)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                in_tensor = cur_layer_input[:, t, :, :, :]
                h_temp = self.pcd_h(in_tensor, h)
                c_temp = self.pcd_c(in_tensor, c)
                h, c = self.cell_list[layer_idx](input_tensor=in_tensor, cur_state=[h_temp, c_temp])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size):
        return super()._init_hidden(batch_size, tensor_size)


class BiDeformableConvLSTM(nn.Module):

    def __init__(self,
                 input_size,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 num_layers,
                 front_RBs,
                 groups,
                 batch_first=False,
                 bias=True,
                 return_all_layers=False):
        super(BiDeformableConvLSTM, self).__init__()
        self.forward_net = DeformableConvLSTM(input_size=input_size,
                                              input_dim=input_dim,
                                              hidden_dim=hidden_dim,
                                              kernel_size=kernel_size,
                                              num_layers=num_layers,
                                              front_RBs=front_RBs,
                                              groups=groups,
                                              batch_first=batch_first,
                                              bias=bias,
                                              return_all_layers=return_all_layers)
        self.conv_1x1 = nn.Conv2d(2 * input_dim, input_dim, 1, 1, bias=True)

    def forward(self, x):
        reversed_idx = list(reversed(range(x.shape[1])))
        x_rev = x[:, reversed_idx, ...]
        out_fwd, _ = self.forward_net(x)
        out_rev, _ = self.forward_net(x_rev)
        rev_rev = out_rev[0][:, reversed_idx, ...]
        B, N, C, H, W = out_fwd[0].size()
        result = torch.cat((out_fwd[0], rev_rev), dim=2)
        result = result.view(B * N, -1, H, W)
        result = self.conv_1x1(result)
        return result.view(B, -1, C, H, W)


class ZSM_encoder(nn.Module):

    def __init__(self, channel):
        super(ZSM_encoder, self).__init__()
        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=channel)
        self.conv_first = nn.Conv2d(3, channel, 3, 1, 1, bias=True)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, 5)
        self.fea_L2_conv1 = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=channel, groups=8)
        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, bias=True)
        self.ConvBLSTM = BiDeformableConvLSTM(input_size=(64,112), input_dim=channel, hidden_dim=[channel], \
                                              kernel_size=(3, 3), num_layers=1, batch_first=True, front_RBs=5,
                                              groups=8)
        #### reconstruction
        self.recon_trunk = mutil.make_layer(ResidualBlock_noBN_f, 40)

    def forward(self, x, target_t):
        B, N, C, H, W = x.size()  # N input video frames
        #### extract LR features
        # L1
        L1_fea = nn.LeakyReLU(negative_slope=0.1, inplace=True)(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = nn.LeakyReLU(negative_slope=0.1, inplace=True)(self.fea_L2_conv1(L1_fea))
        L2_fea = nn.LeakyReLU(negative_slope=0.1, inplace=True)(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = nn.LeakyReLU(negative_slope=0.1, inplace=True)(self.fea_L3_conv1(L2_fea))
        L3_fea = nn.LeakyReLU(negative_slope=0.1, inplace=True)(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### align using pcd
        to_lstm_fea = []
        for idx in range(N - 1):
            fea1 = [L1_fea[:, idx, :, :, :].clone(), L2_fea[:, idx, :, :, :].clone(), L3_fea[:, idx, :, :, :].clone()]
            fea2 = [L1_fea[:, idx + 1, :, :, :].clone(), L2_fea[:, idx + 1, :, :, :].clone(), L3_fea[:, idx + 1, :, :, :].clone()]
            '''aligned_fea = self.pcd_align(fea1, fea2,
                        target_t, 
                        1-target_t)'''

            aligned_fea = self.pcd_align(fea1, fea2, None, None)

            fusion_fea = self.fusion(aligned_fea)  # [B, N, C, H, W]
            if idx == 0:
                to_lstm_fea.append(fea1[0])
            to_lstm_fea.append(fusion_fea)
            to_lstm_fea.append(fea2[0])
        lstm_feats = torch.stack(to_lstm_fea, dim=1)
        #### align using bidirectional deformable conv-lstm
        feats = self.ConvBLSTM(lstm_feats.view(B, N * 2 - 1, -1, H, W))
        B, T, C, H, W = feats.size()

        feats = feats.view(B * T, C, H, W)
        out = self.recon_trunk(feats)
        ###############################################
        out = out.view(B, T, 64, H, W)      # torch.Size([24, 3, 64, 32, 32])
        return out


class LunaTokis(nn.Module):

    def __init__(self, setting=5):
        super(LunaTokis, self).__init__()
        #args = argparse.ArgumentParser().parse_args()
        args, unknown = argparse.ArgumentParser().parse_known_args()
        args.small = True
        args.mixed_precision = False
        args.alternate_corr = True

        ### load raft
        self.flow_predictor = RAFT(args)
        ckpt = torch.load("/saved_checkpoints/bfstvsr.pth") # ['model']
        keys = list(ckpt.keys())
        for key in keys:
            tmp = key.replace("flow_predictor.", "")
            ckpt[tmp] = ckpt[key]
            del ckpt[key]
        self.flow_predictor.load_state_dict(ckpt, strict=True)

        ### warper
        self.fwarp = Softsplat()
        self.fwarp_max = Softsplat_Max()
        self.fwarp_count = Softsplat_Count()
        self.bwarp = BackWarp(clip=True)

        self.unfold = True
        self.norm_gamma = nn.Parameter(torch.ones(1, 3, 1), requires_grad=True)
        self.norm_beta = nn.Parameter(torch.zeros(1, 3, 1), requires_grad=True)

        ### Guassian kernel
        self.g_filter = nn.Parameter(torch.cuda.FloatTensor([[1. / 16., 1. / 8., 1. / 16.], [1. / 8., 1. / 4., 1. / 8.],
                                                             [1. / 16., 1. / 8., 1. / 16.]]).reshape(1, 1, 1, 3, 3),
                                     requires_grad=False)

        ### model setting
        channel = 64
        self.groups = 1  #???
        self.trans = False  # trans for flow_process
        self.siren = True  # Decoder is siren or not
        self.res_liff = False  #LIIF residual
        self.local_ensemble = False  #LIIF

        self.warp_to_many = False if setting < 6 else True  # warp to two frames or not
        self.decoder_Z = False if setting < 5 else True  # decoder input have z or not
        self.predict_Z = False if setting < 4 else True
        self.input_Z = False if setting < 3 else True
        self.predict_flow = False if setting < 2 else True

        ### modules
        '''
            encoder: ZSM
            imnet: SINF, Siren
            flow_imnet : STINF, Siren
            synth_net : Decoder, Siren
            flow_process: flow encoder
		'''
        self.encoder = ZSM_encoder(channel)
        self.flow_imnet = Siren(in_features=67, out_features=3 * self.groups, hidden_features=[64, 64, 256], hidden_layers=2, outermost_linear=True)
        self.imnet = Siren(in_features=66, out_features=64, hidden_features=[64, 64, 256], hidden_layers=2, outermost_linear=True)
        if self.res_liff:
            self.res_imnet = Siren(in_features=66, out_features=64, hidden_features=[64, 64, 256], hidden_layers=2, outermost_linear=True)
        if not self.siren:
            self.synth_net = nn.Sequential(
                nn.Conv2d(192 + 5 * self.groups + 64 * self.res_liff + self.decoder_Z + self.warp_to_many * 133,
                          channel * 2,
                          3,
                          1,
                          1,
                          bias=True,
                          groups=1), nn.Conv2d(channel * 2, channel * 2, 3, 1, 1, bias=True, groups=1),
                nn.Conv2d(channel * 2, channel, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True), LateralBlock(channel),
                LateralBlock(channel), LateralBlock(channel), LateralBlock(channel), LateralBlock(channel),
                nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(channel, 3, 3, 1, 1, padding_mode='reflect', bias=True))
        else:
            self.synth_net = Siren(in_features=192 + 5 * self.groups + 64 * self.res_liff + self.decoder_Z + self.warp_to_many * 133,
                                   out_features=3,
                                   hidden_features=[64, 64, 64, 256],
                                   hidden_layers=3,
                                   outermost_linear=True)
        # self.trans = False
        if not self.trans:
            self.flow_process = nn.Sequential(nn.Conv2d(8 + self.input_Z * 6, channel, 3, 1, 1, bias=True, groups=2),
                                              nn.Conv2d(channel, channel, 3, 1, 1, bias=True, groups=2), nn.LeakyReLU(negative_slope=0.1,
                                                                                                                      inplace=True),
                                              LateralBlock(channel), LateralBlock(channel), LateralBlock(channel), LateralBlock(channel),
                                              LateralBlock(channel), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              nn.Conv2d(channel, channel, 3, 1, 1, padding_mode='reflect', bias=True))
        else:
            self.flow_process = nn.Sequential(nn.Conv2d(4 + self.input_Z * 3, channel // 2, 3, 1, 1, bias=True, groups=1),
                                              nn.Conv2d(channel // 2, channel, 3, 1, 1, bias=True, groups=1),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True), LateralBlock(channel), LateralBlock(channel),
                                              LateralBlock(channel), LateralBlock(channel), LateralBlock(channel),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              nn.Conv2d(channel, channel, 3, 1, 1, padding_mode='reflect', bias=True))
            self.weight_generator = nn.Sequential(LateralBlock(channel), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                                  nn.Conv2d(channel, self.groups, 3, 1, 1, padding_mode='reflect', bias=True))
        self.alpha = nn.Parameter(torch.ones(1) * -20., requires_grad=True)
        self.shuffle = nn.Conv2d(channel, channel, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, input_target_frames, target_t, scale=None, rank=0, train_idx=0, use_GT=True, iter=12, flows=None):

        x_for_encoder = x.clone()     
        x = x.permute(0, 2, 1, 3, 4)
        x = x[:, :, x.shape[2] // 2 - 1:x.shape[2] // 2 + 1]

        ### generate motion and reliability maps
        with torch.no_grad():
            target_t = torch.stack(target_t, 1).squeeze(-1)     # torch.Size([B, 1])
            B, N = target_t.shape
            B, _, _, H, W = x.shape

            # calculate HR size
            if type(scale) == type([]):
                HH, WW = scale[0][0], scale[1][0]
                # print("HH, WW : ", HH, WW)
            else:
                HH, WW = round(H * scale), round(W * scale)

            fr0, fr1 = x[:, :, 0], x[:, :, 1]

            if False:
                flow, psies, flow_GT = flows
                flow = flow.permute(1, 0, 2, 3, 4).reshape(2, B, 2, H, W)
                flow = torch.cat([torch.zeros((1, B, 2, H, W), device=flow.device), flow, torch.zeros((1, B, 2, H, W), device=flow.device)], 0)
                psies = psies.permute(1, 0, 2, 3, 4).reshape(4 * B, 3, H, W)
                #flow_GT = flow_GT.permute(1,0,2,3,4).reshape(2*N*B,2,HH,WW)
            else:
                # generate HR input motion
                x_norm = interpolate(x.reshape(B, -1, H, W), size=(HH, WW), mode='bilinear', align_corners=False).reshape(B, -1, 2, HH, WW) 
                fr0, fr1 = x_norm[:, :, 0], x_norm[:, :, 1]

                # use pretrained raft generate HR motion 00,01,10,11
                flow = self.flow_predictor(torch.cat([fr0, fr0, fr1, fr1], dim=0) * 255., torch.cat([fr0, fr1, fr0, fr1], dim=0) * 255., iters=iter)[-1]
                fr0, fr1 = x[:, :, 0], x[:, :, 1]  
                # downsample motion
                flow = interpolate(flow, size=(H, W), mode='bilinear', align_corners=False) * ((H) / (HH)) 

                # reshape, replace motion 00 and 11 into zero
                flow = flow.reshape(4, B, 2, H, W)
                flow[0] *= 0.
                flow[3] *= 0.
                flow = flow.reshape(4 * B, 2, H, W)
                flow /= 1.

                # compute Z importance metric
                '''
                psi_photo: use estimated flow to warp frame and calculate l1 loss
                psi_flow: use flow to warp flow and caclute l1 loss
                '''
                warped, _ = self.bwarp(torch.cat([fr0, fr1, fr0, fr1], dim=0), flow)
                psi_photo = torch.nn.functional.l1_loss(input=torch.cat([fr0, fr0, fr1, fr1], dim=0), target=warped, reduction='none').mean(1) 
                flow = flow.reshape(4, B, 2, H, W)    
                warped, _ = self.bwarp(-torch.cat([
                    flow[0],
                    flow[2],
                    flow[1],
                    flow[3],
                ], dim=0), flow.reshape(4 * B, 2, H, W))        
                psi_flow = torch.nn.functional.l1_loss(input=(flow.reshape(4 * B, 2, H, W)), target=warped, reduction='none').mean(1)
                f = (flow).reshape(4 * B, -1, H, W)
                sqaure_mean, mean_square = torch.split(F.conv3d(
                    F.pad(torch.cat([f**2, f], 1), (1, 1, 1, 1), mode='reflect').unsqueeze(1), self.g_filter).squeeze(1),
                                                       2,
                                                       dim=1)
                psi_var = (sqaure_mean - mean_square**2).clip(1e-9, None).sqrt().mean(1)
                psies = torch.stack([psi_photo, psi_flow / 10., psi_var], dim=1) 

                flow_GT = 0
                #if (flows is not None) and use_GT:
                #    # hr_gt_flow B N 4 2 HH WW
                #    flow_GT = hr_gt_flow.permute(2,0,1,3,4,5).reshape(2*B*N,2,HH,WW)
                #else:

            # flow teacher forcing needs GT flow
            if self.training:
                input_target_frames = interpolate(input_target_frames.reshape(B, -1, HH, WW), size=(128, 128), mode='bilinear',
                                                  align_corners=False).reshape(B, -1, 3, 128, 128)        
                t_fr0, t_fr1 = input_target_frames[:, 0], input_target_frames[:, -1],
                t_frs = input_target_frames[:, 1:-1]
                flow_GT = self.flow_predictor(torch.cat([t_fr0, t_fr1], dim=0).repeat(1, N, 1, 1).reshape(2 * B * N, 3, 128, 128) * 255.,
                                              t_frs.unsqueeze(0).repeat(2, 1, 1, 1, 1, 1).reshape(2 * B * N, 3, 128, 128) * 255.,
                                              iters=iter)[-1]     
                flow_GT = interpolate(flow_GT, size=(HH, WW), mode='bilinear', align_corners=False) * (HH / 128)     

        ### generate spatial and flow feature with encoder
        pyramid = [None]
        #feat = self.encoder(f_lv.repeat(N,1,1,1,1), target_t.permute(1,0).reshape(-1,1,1,1))
        #feat = self.encoder(x_for_encoder, None)
        feat = self.encoder(torch.stack([fr0, fr1], 1), None)       # torch.Size([B, 3, 64, 32, 32])
        #  feat B 7 C H W
        #residual = feat[:,1].reshape(N,B,-1,H,W).permute(1,0,2,3,4).reshape(B*N,-1,H,W)
        '''
        feat: F_0_L, F_1_L
        residual: F_01_L
        flow_feat: [flow, psies, ref_start_durations]
	    '''
        residual = feat[:, feat.shape[1] // 2].reshape(B, -1, H, W)     
        #feat = feat.reshape(N,B,3,-1,H,W).permute(1,0,2,3,4,5).reshape(B*N,3,-1,H,W)
        feat = torch.cat((feat[:, feat.shape[1] // 2 - 1], feat[:, feat.shape[1] // 2 + 1]), 0)  
        #feat = self.shuffle(feat)
        pyramid.append(feat)
        flow = flow.reshape(4 * B, 2, H, W)   
        ref_start_durations = [
            [0, 0],
            [0, 8],
            [8, 0],
            [8, 8],
        ]
        ref_start_durations = torch.cuda.FloatTensor(ref_start_durations, device=target_t.device).unsqueeze(1)

        # self.trans = False
        if not self.trans:
            if self.input_Z:
                flow_feat = torch.cat((
                    (flow / 20.).reshape(2, 2, B, -1, H, W).permute(0, 2, 1, 3, 4, 5).reshape(2 * B, 2, -1, H, W),                  
                    psies.reshape(2, 2, B, -1, H, W).permute(0, 2, 1, 3, 4, 5).reshape(2 * B, 2, -1, H, W),                         
                    ref_start_durations.reshape(2, 4, 1, 1).unsqueeze(1).repeat(1, B, 1, H, W).reshape(2 * B, 2, 2, H, W) / 8.0,    
                ),
                                      dim=2).reshape(2 * B, -1, H, W)
            else:
                flow_feat = torch.cat((
                    (flow / 20.).reshape(2, 2, B, -1, H, W).permute(0, 2, 1, 3, 4, 5).reshape(2 * B, 2, -1, H, W),
                    ref_start_durations.reshape(2, 4, 1, 1).unsqueeze(1).repeat(1, B, 1, H, W).reshape(2 * B, 2, 2, H, W) / 8.0,
                ),
                                      dim=2).reshape(2 * B, -1, H, W)
            flow_feat = self.flow_process(flow_feat)                   
        else:
            if self.input_Z:
                flow_feat = torch.cat((
                    (flow / 20.).reshape(4 * B, -1, H, W),
                    psies.reshape(4 * B, -1, H, W),
                    ref_start_durations.reshape(4, 2, 1, 1).repeat(1, B, H, W).reshape(4 * B, 2, H, W) / 8.0,
                ),
                                      dim=1)
            else:
                flow_feat = torch.cat((
                    (flow / 20.).reshape(4 * B, -1, H, W),
                    ref_start_durations.reshape(4, 2, 1, 1).repeat(1, B, H, W).reshape(4 * B, 2, H, W) / 8.0,
                ),
                                      dim=1)
            flow_feat = self.flow_process(flow_feat)
            weight = self.weight_generator(flow_feat).reshape(4, 4, B, self.groups, 1, H, W)
            flow_feat = flow_feat.reshape(4, 4, B, self.groups, -1, H, W)
            weight = nn.functional.softmax(weight, 1)
            flow_feat = (flow_feat * weight).sum(1).reshape(4 * B, -1, H, W)

        ### LIIF (SINF & STINF): generate spatial and flow feature with encoder
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        hr_coord = make_coord((HH, WW)).unsqueeze(0)        
        hr_coord = hr_coord.to(flow_feat.device)
        eps_shift = 1e-6

        # warpped_feat = feat (F_0_L, F_1_L)
        warpped_feat = pyramid[1]       

        rx = 2 / H / 2
        ry = 2 / W / 2

        feat_coord = make_coord((H,W), flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(1, 2,H,W)      

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:

                coord_ = hr_coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                c1, c3, c4, c5 = 2 * B * warpped_feat.shape[1], 2 * B * flow_feat.shape[1], feat_coord.shape[1], residual.shape[1] * B
                '''
                to_be_warp (LR)
                    warpped_feat = LR feat (F_0_L, F_1_L)
                    flow_feat = LR output of flow encoder (T_0_L, T_1_L)
                    feat_coord = LR coord
                    residual = LR F_01_L
                '''
                to_be_warp = torch.cat(
                    (warpped_feat.reshape(1, c1, H, W), flow_feat.reshape(1, c3, H, W), feat_coord.reshape(1, c4, H, W), residual.reshape(
                        1, c5, H, W)), 1).reshape(1, -1, H, W)       

                # upsample with grid_sample
                warped = F.grid_sample(to_be_warp, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :]    
                '''
                q_feat = HR feat (F_0_L, F_1_L)
                q_flow_feat = HR output of flow encoder (T_0_L, T_1_L)
                q_coord = HR coord
                q_residual = HR F_01_L
                '''
                q_feat, warped = warped[:, :c1], warped[:, c1:]
                q_flow_feat, warped = warped[:, :c3], warped[:, c3:]
                q_coord, warped = warped[:, :c4], warped[:, c4:]
                q_residual, warped = warped[:, :c5], warped[:, c5:]
                q_feat = q_feat.reshape(2 * B, -1, HH * WW).permute(0, 2, 1)                
                q_flow_feat = q_flow_feat.reshape(2 * B, -1, HH * WW).permute(0, 2, 1)      
                q_coord = q_coord.reshape(1, -1, HH * WW).permute(0, 2, 1)
                q_residual = q_residual.reshape(B, -1, HH * WW).permute(0, 2, 1)

                rel_coord = hr_coord - q_coord
                rel_coord[:, :, 0] *= H
                rel_coord[:, :, 1] *= W
                qs = rel_coord.shape[1]
                q_feat_low = q_feat.clone()

                # assign input for q_flow_feat and q_feat
                q_flow_feat, q_feat  = torch.cat([q_flow_feat.repeat(1,N,1).reshape(2*B*N,qs,-1),
                                                target_t.reshape(B*N,1,1).repeat(2,qs,1),
                                                rel_coord.repeat(2*B*N,1,1)
                                                ], dim=-1), \
                                    torch.cat([q_feat.reshape(2*B,qs,-1),
                                                rel_coord.repeat(2*B,1,1),
                                                ], dim=-1)

                # predict with imnet and flow_imnet
                q_flow_feat = self.flow_imnet(q_flow_feat)      
                q_feat = self.imnet(q_feat)                     

                if self.res_liff:
                    q_residual_low = q_residual.clone()
                    q_residual = torch.cat([
                        q_residual.reshape(B, qs, -1),
                        rel_coord.repeat(B, 1, 1),
                    ], dim=-1)
                    q_residual = self.res_imnet(q_residual)
                    preds.append([q_feat, q_feat_low, q_residual, q_flow_feat, q_residual_low])
                else:
                    preds.append([
                        q_feat,
                        q_feat_low,
                        q_residual,
                        q_flow_feat,
                    ])
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        if self.res_liff:
            ret = [0, 0, 0, 0, 0]
        else:
            ret = [0, 0, 0, 0]
        for pred, area in zip(preds, areas):
            for i, feat in enumerate(pred):
                ret[i] = ret[i] + feat * (area / tot_area).unsqueeze(-1)
        if self.res_liff:
            q_feat, q_feat_low, q_residual, q_flow_feat, q_residual_low = ret
        else:
            q_feat, q_feat_low, q_residual, q_flow_feat = ret         

        feat = q_feat.reshape(2 * B, HH, WW, -1).permute(0, 3, 1, 2)            
        feat_low = q_feat_low.reshape(2 * B, HH, WW, -1).permute(0, 3, 1, 2)    
        q_residual = q_residual.reshape(B, HH, WW, -1).permute(0, 3, 1, 2)
        if self.res_liff:
            q_residual_low = q_residual_low.reshape(B, HH, WW, -1).permute(0, 3, 1, 2)
            # q_residual_low = q_residual_low.reshape(B * N, HH, WW, -1).permute(0, 3, 1, 2)
            q_residual = torch.cat([q_residual, q_residual_low], 1)
        flow = q_flow_feat.reshape(2 * B * N, HH, WW, -1).permute(0, 3, 1, 2).reshape(2 * B * N * self.groups, -1, HH, WW)  
        #if use_GT:
        #if False:
        if self.training:
            flow_GT = flow_GT.repeat(1, self.groups, 1, 1).reshape(2 * B * N * self.groups, -1, HH, WW)
        feat = torch.cat([
            feat.reshape(2 * B, -1, HH, WW).repeat(1, N, 1, 1).reshape(2 * B * N * self.groups, -1, HH, WW), flow[:, :-1].detach(),
            feat_low.reshape(2 * B, -1, HH, WW).repeat(1, N, 1, 1).reshape(2 * B * N * self.groups, -1, HH, WW)
        ], 1)
        
        # assign HR flow and z
        flow, z = flow[:, :-1] * 20. * (HH / H), (nn.ReLU()(flow[:, -1].unsqueeze(1)) * self.alpha)
        if not self.predict_Z:
            z = torch.zeros_like(z)

        
        if use_GT:
            output, warped_z = self.fwarp(feat, flow_GT, z)   
            z_max = self.fwarp_max(z.exp(), flow_GT).detach()  
            count = self.fwarp_count(z_max, flow_GT).detach()  
        else:
            output, warped_z = self.fwarp(feat, flow, z)
            z_max = self.fwarp_max(z.exp(), flow).detach()
            count = self.fwarp_count(z, flow).detach()
            
        
        ###????
        if not self.warp_to_many:
            output = output.reshape(2, B * N * self.groups, -1, HH, WW).sum(0)  
            warped_z = warped_z.reshape(2, B * N * self.groups, -1, HH, WW).sum(0) 
            warped_z[warped_z == 0] = 1.0
            output /= warped_z
            z_max = z_max.reshape(2, B * N * self.groups, -1, HH, WW).max(0)[0]
            count = count.reshape(2, B * N * self.groups, -1, HH, WW).sum(0)
            
        else:
            output = output.reshape(2 * B * N * self.groups, -1, HH, WW)
            warped_z = warped_z.reshape(2 * B * N * self.groups, -1, HH, WW)
            warped_z[warped_z == 0] = 1.0
            output /= warped_z
            z_max = z_max.reshape(2 * B * N * self.groups, -1, HH, WW)
            count = count.reshape(2 * B * N * self.groups, -1, HH, WW)

        ###????
        count_ = count.clone()
        count_[count_ == 0.0] = 1.0
        warped_z_ = warped_z.clone()
        warped_z_[warped_z_ == 1.0] = 0.0
        
        
        if self.decoder_Z:
            extra = torch.cat((z_max, count / 16., (warped_z_ / count_)), 1)
        else:
            extra = torch.cat((count / 16., (warped_z_ / count_)), 1)

        if not self.warp_to_many:
            output_all = torch.cat((
                output.reshape(B * N, -1, HH, WW),
                extra.reshape(B * N, -1, HH, WW),
                q_residual.repeat(1, N, 1, 1).reshape(B * N, -1, HH, WW),
                target_t.reshape(B * N, 1, 1, 1).repeat(1, 1, HH, WW),
            ), 1).reshape(B * N, -1, HH, WW)
        else:
            output_all = torch.cat((
                output.reshape(2, B * N, -1, HH, WW).permute(1, 0, 2, 3, 4).reshape(B * N, -1, HH, WW),
                extra.reshape(2, B * N, -1, HH, WW).permute(1, 0, 2, 3, 4).reshape(B * N, -1, HH, WW),
                q_residual.repeat(1, N, 1, 1).reshape(B * N, -1, HH, WW),
                target_t.reshape(B * N, 1, 1, 1).repeat(1, 1, HH, WW),
            ), 1).reshape(B * N, -1, HH, WW)        # torch.Size([B, 198, 128, 128])
        if not self.siren:
            output_all = self.synth_net(output_all).reshape(B, N, -1, HH, WW).permute(1, 0, 2, 3, 4)
        else:
            output_all = self.synth_net(output_all.reshape(B * N, -1, HH * WW).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, N, -1, HH, WW).permute(1, 0, 2, 3, 4)

        return torch.clamp((output_all), 0, 1), flow / 20.0 / (HH / H), (flow_GT / 20.0) / (HH / H)     # torch.Size([1, B, 3, 128, 128])


class LateralBlock(nn.Module):

    def __init__(self, dim):
        super(LateralBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(dim, dim, 3, 1, 1, bias=True))

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class BackWarp(nn.Module):

    def __init__(self, clip=True):
        super(BackWarp, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip = clip

    def forward(self, img, flow, mode='bilinear'):
        b, c, h, w = img.shape      
        b, c, h, w = flow.shape    
        gridY, gridX = torch.meshgrid(torch.arange(h), torch.arange(w))     
        gridX, gridY = gridX.to(img.device, non_blocking=True), gridY.to(img.device, non_blocking=True)

        u = flow[:, 0]  # W
        v = flow[:, 1]  # H

        x = repeat(gridX, 'h w -> b h w', b=b).float() + u
        y = repeat(gridY, 'h w -> b h w', b=b).float() + v

        # normalize
        x = (x / w) * 2 - 1
        y = (y / h) * 2 - 1

        # stacking X and Y
        grid = torch.stack((x, y), dim=-1)

        # Sample pixels using bilinear interpolation.
        if self.clip:
            output = grid_sample(img, grid, mode=mode, align_corners=True, padding_mode='border')
        else:
            output = grid_sample(img, grid, mode=mode, align_corners=True)
        return output, grid
