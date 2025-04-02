import os
from glob import glob

import torch

from metrics import basicMetric as M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Metrics:
    def __init__(self, metrics=[], skip_ref_frames=2, batch_size=1) -> None:
        for metric in metrics:
            if metric == "flolpips":
                print("Building FLOLPIPS metric...")
                from metrics.flolpips import calc_flolpips

                self.FloLPIPS = calc_flolpips
            elif metric == "vfips":
                print("Building VFIPS metric...")
                from metrics.VFIPS import calc_vfips

                self.VFIPS = calc_vfips
            else:
                raise ValueError("metric %s is not supported." % metric)

        self.metrics = metrics
        self.skip = skip_ref_frames

    def _filter_ref_frames_(self, images: list):
        if isinstance(self.skip, int):
            del images[:: self.skip]

    def calc_flolpips(self, meta):
        tmpDir = None # meta["tmpDir"]
        disMP4 = meta["disMP4"]
        refMP4 = meta["refMP4"]
        hwt = meta["hwt"]
        score = self.FloLPIPS(disMP4, refMP4, tmpDir, hwt)
        return score

    def calc_vfips(self, meta):
        dis = meta["disImgs"]
        ref = meta["refImgs"]
        score = self.VFIPS(dis_dir=dis, ref_dir=ref)
        return abs(score)

    def eval(self, meta):
        result = {}
        for metric in self.metrics:
            result[metric] = self.__getattribute__("calc_%s" % metric)(meta)
        return result
