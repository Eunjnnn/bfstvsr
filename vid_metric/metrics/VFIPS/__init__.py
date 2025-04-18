import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from . import networks
from .utils import *


def calc_vfips(dis_dir, ref_dir):
    moduleNetwork = networks.get_model("multiscale_v33")
    moduleNetwork.load_state_dict(torch.load("./checkpoints/VFIPS.pytorch"))    # https://github.com/hqqxyy/VFIPS/tree/main/exp/eccv_ms_multiscale_v33

    moduleNetwork.cuda().eval()

    @torch.no_grad()
    def estimate(tenRef, tenVideo):
        tenDis = moduleNetwork(tenRef, tenVideo)
        return tenDis

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(transform_list)

    disLst = sorted(os.listdir(dis_dir))
    refLst = sorted(os.listdir(ref_dir))

    frame_count = len(disLst)

    # stride 12
    # for start_id in range(0, frame_count - 12):
    scores = []
    for start_id in range(0, frame_count - 12, 12):
        video = []
        gt = []
        for i in range(12):
            videoimg = cv2.imread(os.path.join(dis_dir, disLst.pop(0)))
            videoimg = cv2.cvtColor(videoimg, cv2.COLOR_BGR2RGB)
            videoimg = Image.fromarray(videoimg)
            videoimg = transform(videoimg).unsqueeze(0)

            gtimg = cv2.imread(os.path.join(ref_dir, refLst.pop(0)))
            gtimg = cv2.cvtColor(gtimg, cv2.COLOR_BGR2RGB)
            gtimg = Image.fromarray(gtimg)
            gtimg = transform(gtimg).unsqueeze(0)

            video.append(videoimg)
            gt.append(gtimg)

        video = torch.cat(video, dim=0)
        gt = torch.cat(gt, dim=0)

        video = video.unsqueeze(0).cuda()
        gt = gt.unsqueeze(0).cuda()

        dis = estimate(gt, video)
        dis = dis.data.cpu().numpy().flatten()

        scores.append(dis)
    return np.mean(scores)
