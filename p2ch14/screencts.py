import argparse
import sys

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from util.util import enumerateWithEstimate, prhist
from .dsets import getCandidateInfoList, getCtSize, getCt
from util.logconf import logging
# from .model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class LunaScreenCtDataset(Dataset):
    def __init__(self):
        self.series_list = sorted(set(candidateInfo_tup.series_uid for candidateInfo_tup in getCandidateInfoList()))

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, ndx):
        series_uid = self.series_list[ndx]
        ct = getCt(series_uid)
        mid_ndx = ct.hu_a.shape[0] // 2

        air_mask, lung_mask, dense_mask, denoise_mask, tissue_mask, body_mask, altneg_mask = ct.build2dLungMask(mid_ndx)

        return series_uid, float(dense_mask.sum() / denoise_mask.sum())


class LunaScreenCtApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        # parser.add_argument('--scaled',
        #     help="Scale the CT chunks to square voxels.",
        #     default=False,
        #     action='store_true',
        # )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        self.prep_dl = DataLoader(
            LunaScreenCtDataset(),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        series2ratio_dict = {}

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Screening CTs",
            start_ndx=self.prep_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            series_list, ratio_list = batch_tup
            for series_uid, ratio_float in zip(series_list, ratio_list):
                series2ratio_dict[series_uid] = ratio_float
            # break

        prhist(list(series2ratio_dict.values()))




if __name__ == '__main__':
    LunaScreenCtApp().main()
