import os
import os.path as osp

import cv2
import lpips
import numpy as np
import torch
from DISTS_pytorch import DISTS
from pytorch_msssim import ms_ssim, ssim
from torchvision import transforms
from tqdm import tqdm


def psnr(pred, gt, data_range=1.0, size_average=False):
    diff = (pred - gt).div(data_range)
    mse = diff.pow(2).mean(dim=(-3, -2, -1))
    psnr = -10 * torch.log10(mse + 1e-8)
    if size_average:
        return torch.mean(psnr)
    else:
        return psnr


def ie(pred, gt, data_range=1.0, size_average=False):
    diff = (pred - gt).div(data_range) * 255.0
    mae = diff.abs().mean(dim=(-3, -2, -1))

    if size_average:
        return torch.mean(mae)
    else:
        return mae


def calc_avg_metric_(func, dis, ref, device="cpu", verbose=0, batch=1):
    if isinstance(dis, str):  # a directory
        disLst = [osp.join(dis, x) for x in sorted(os.listdir(dis))]
        refLst = [osp.join(ref, x) for x in sorted(os.listdir(ref))]
    elif isinstance(dis, (tuple, list)):  # list of image paths
        disLst = dis
        refLst = ref
    frame_count = len(disLst)
    assert len(disLst) == len(refLst), f"{len(disLst)} should be {len(refLst)}"
    scores = []

    TF = transforms.ToTensor()
    batch_dis = []
    batch_ref = []
    gen = tqdm(range(frame_count)) if verbose else range(frame_count)

    for i in gen:
        disImg = cv2.imread(disLst[i])
        disImg = cv2.cvtColor(disImg, cv2.COLOR_BGR2RGB)
        disTen = TF(disImg).unsqueeze(0).to(device)

        refImg = cv2.imread(refLst[i])
        refImg = cv2.cvtColor(refImg, cv2.COLOR_BGR2RGB)
        refTen = TF(refImg).unsqueeze(0).to(device)

        batch_dis.append(disTen)
        batch_ref.append(refTen)
        if i % batch == 0 or i == frame_count - 1:
            score = func(torch.cat(batch_dis), torch.cat(batch_ref))
            score = score.data.cpu().numpy().flatten()
            scores.append(score)
            batch_dis = []
            batch_ref = []

    all_score = np.concatenate(scores)
    avg = np.mean(all_score)
    if verbose:
        print("AVG %s: " % func.__name__.upper(), avg)
    return avg


class basicMetric:
    def __init__(self) -> None:
        pass

    @staticmethod
    def calc_ssim(dis, ref, **kwargs):
        def ssim_(a, b):
            return ssim(a, b, data_range=1.0, size_average=False)

        return calc_avg_metric_(ssim_, dis, ref, **kwargs)

    @staticmethod
    def calc_msssim(dis, ref, **kwargs):
        def ms_ssim_(a, b):
            return ms_ssim(a, b, data_range=1.0, size_average=False)

        return calc_avg_metric_(ms_ssim_, dis, ref, **kwargs)

    @staticmethod
    def calc_psnr(dis, ref, **kwargs):
        def psnr_(a, b):
            return psnr(a, b, data_range=1.0, size_average=False)

        return calc_avg_metric_(psnr_, dis, ref, **kwargs)

    @staticmethod
    def calc_lpips(dis, ref, **kwargs):
        model = lpips.LPIPS(net="alex", verbose=False)
        model.to(kwargs["device"]).eval()

        def lpips_(a, b):
            with torch.no_grad():
                score = model.forward(a, b, normalize=True)
            return score

        return calc_avg_metric_(lpips_, dis, ref, **kwargs)

    @staticmethod
    def calc_dists(dis, ref, **kwargs):
        model = DISTS()
        model.to(kwargs["device"]).eval()

        def lpips_(a, b):
            with torch.no_grad():
                score = model.forward(a, b)
            return score

        return calc_avg_metric_(lpips_, dis, ref, **kwargs)

    @staticmethod
    def calc_ie(dis, ref, **kwargs):
        def ie_(out, gt):
            return ie(out, gt, data_range=1.0, size_average=False)

        return calc_avg_metric_(ie_, dis, ref, **kwargs)
