# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import pywt
import basicsr.models.losses.SWT as SWT

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class TotalLoss(nn.Module):
    def __init__(self, loss_weight=0.01, reduction='mean'):
        super(TotalLoss, self).__init__()
        self.loss1 = MSELoss()
        # self.loss2 = FrequencyCharLoss()
        self.loss_weight = loss_weight

    def forward(self,pred, target):
        
        loss1 = self.loss1(pred[0], target)
        # loss2 = self.loss2(pred, target)
        # print(len(pred))
        return loss1 + self.loss_weight * pred[1]

class ComposeLoss(nn.Module):
    def __init__(self, loss_weight=0.01, reduction='mean'):
        super(ComposeLoss, self).__init__()
        self.loss1 = MSELoss()
        self.loss2 = FrequencyLoss()
        self.loss_weight = loss_weight

    def forward(self,pred, target):
        loss1 = self.loss1(pred, target)
        loss2 = self.loss2(pred, target)
        # print(loss1, loss2)

        return loss1+self.loss_weight*loss2


class FrequencyCharLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-12, reduction='mean'):
        super(FrequencyCharLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        diff_real = torch.sqrt((pred_fft.real - target_fft.real)**2 + self.eps)
        diff_imag = torch.sqrt((pred_fft.imag - target_fft.imag)**2 + self.eps)
        freq_distance = diff_real + diff_imag
        return self.loss_weight*torch.mean(freq_distance)

class FrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-12, reduction='mean'):
        super(FrequencyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.cri = nn.L1Loss(reduction=reduction)
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        diff_real = self.cri(pred_fft.real , target_fft.real)
        diff_imag = self.cri(pred_fft.imag , target_fft.imag)
        freq_distance = diff_real + diff_imag
        return self.loss_weight*freq_distance


class WaveletLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(WaveletLoss, self).__init__()
        # wavelet init
        self.l_pix_w = 0.05
        self.l_pix_w_lh = 0.02
        self.l_pix_w_hl = 0.02
        self.l_pix_w_hh = 0.05
        filter_name = "sym7"
        wavelet = pywt.Wavelet(filter_name)
        self.cri_pix = nn.L1Loss()
        self.eps = eps
            
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2*np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        self.sfm = SWT.SWTForward(1, filters, 'periodic')
        # self.ifm = SWT.SWTInverse(filters, 'periodic')  

    def forward(self, pred, target, **kwargs):
        ## wavelet bands of sr image
        sr_img_y       = 16.0 + (pred[:,0:1,:,:]*65.481 + pred[:,1:2,:,:]*128.553 + pred[:,2:,:,:]*24.966)
        wavelet_sr     = self.sfm(sr_img_y)[0]

        self.LL_band   = wavelet_sr[:,0:1, :, :]
        self.LH_band   = wavelet_sr[:,1:2, :, :]
        self.HL_band   = wavelet_sr[:,2:3, :, :]
        self.HH_band   = wavelet_sr[:,3:, :, :]

        self.combined_HF_bands     = torch.cat((self.LH_band, self.HL_band, self.HH_band), axis = 1)           

        ## wavelet bands of hr image
        hr_img_y       = 16.0 + (target[:,0:1,:,:]*65.481 + target[:,1:2,:,:]*128.553 + target[:,2:,:,:]*24.966)
        wavelet_hr     = self.sfm(hr_img_y)[0]

        self.LL_band_hr   = wavelet_hr[:,0:1, :, :]
        self.LH_band_hr   = wavelet_hr[:,1:2, :, :]
        self.HL_band_hr   = wavelet_hr[:,2:3, :, :]
        self.HH_band_hr   = wavelet_hr[:,3:, :, :]

        self.combined_HF_bands_hr     = torch.cat((self.LH_band_hr, self.HL_band_hr, self.HH_band_hr), axis = 1)       
 

        l_g_total = 0  # pixel loss
        l_g_pix = self.l_pix_w * self.cri_pix(self.LL_band, self.LL_band_hr)
        l_g_pix_lh = self.l_pix_w_lh * self.cri_pix(self.LH_band, self.LH_band_hr)
        l_g_pix_hl = self.l_pix_w_hl * self.cri_pix(self.HL_band, self.HL_band_hr)
        l_g_pix_hh = self.l_pix_w_hh * self.cri_pix(self.HH_band, self.HH_band_hr)
        # l_g_pix = self.l_pix_w * torch.sqrt((self.LL_band - self.LL_band_hr) ** 2 + self.eps)
        # l_g_pix_lh = self.l_pix_w_lh * torch.sqrt((self.LH_band - self.LH_band_hr) ** 2 + self.eps)
        # l_g_pix_hl = self.l_pix_w_hl * torch.sqrt((self.HL_band - self.HL_band_hr) ** 2 + self.eps)
        # l_g_pix_hh = self.l_pix_w_hh * torch.sqrt((self.HH_band - self.HH_band_hr) ** 2 + self.eps)
        
        l_g_total = l_g_total  + l_g_pix + l_g_pix_lh + l_g_pix_hl + l_g_pix_hh
        # print("LL: ", l_g_pix, "LH: ", l_g_pix_lh, "HL: ", l_g_pix_hl, "HH: ", l_g_pix_hh)
        # print(l_g_total)

        return l_g_total

        

class TVLoss(nn.Module):
    def __init__(self,loss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = loss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

