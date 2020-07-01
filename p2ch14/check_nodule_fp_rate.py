import argparse
import glob
import hashlib
import math
import os
import sys

import numpy as np
import scipy.ndimage.measurements as measure
import scipy.ndimage.morphology as morph

import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
# from .dsets import LunaDataset, Luna2dSegmentationDataset, getCt, getCandidateInfoList, CandidateInfoTuple
from p2ch13.dsets import Luna2dSegmentationDataset, getCt, getCandidateInfoList, getCandidateInfoDict, CandidateInfoTuple
from p2ch14.dsets import LunaDataset
from p2ch13.model import UNetWrapper
from p2ch14.model import LunaModel

from util.logconf import logging
from util.util import xyz2irc, irc2xyz

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class FalsePosRateCheckApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
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

        parser.add_argument('--series-uid',
            help='Limit inference to this Series UID only.',
            default=None,
            type=str,
        )

        parser.add_argument('--include-train',
            help="Include data that was in the training set. (default: validation data only)",
            action='store_true',
            default=False,
        )

        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default=None,
        )

        parser.add_argument('--classification-path',
            help="Path to the saved classification model",
            nargs='?',
            default=None,
        )

        parser.add_argument('--tb-prefix',
            default='p2ch13',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        self.cli_args = parser.parse_args(sys_argv)
        # self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.initModelPath('seg')

        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.initModelPath('cls')

        self.seg_model, self.cls_model = self.initModels()

    def initModelPath(self, type_str):
        # local_path = os.path.join(
        #     'data-unversioned',
        #     'part2',
        #     'models',
        #     self.cli_args.tb_prefix,
        #     type_str + '_{}_{}.{}.state'.format('*', '*', 'best'),
        # )
        #
        # file_list = glob.glob(local_path)
        # if not file_list:
        pretrained_path = os.path.join(
            'data',
            'part2',
            'models',
            type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
        )
        file_list = glob.glob(pretrained_path)
        # else:
        #     pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([pretrained_path, file_list])
            raise

    def initModels(self):
        with open(self.cli_args.segmentation_path, 'rb') as f:
            log.debug(self.cli_args.segmentation_path)
            log.debug(hashlib.sha1(f.read()).hexdigest())

        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )
        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        with open(self.cli_args.classification_path, 'rb') as f:
            log.debug(self.cli_args.classification_path)
            log.debug(hashlib.sha1(f.read()).hexdigest())

        cls_dict = torch.load(self.cli_args.classification_path)

        cls_model = LunaModel()
        # cls_model = AlternateLunaModel()
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model = seg_model.to(self.device)
            cls_model = cls_model.to(self.device)

        self.conv_list = nn.ModuleList([
            self._make_circle_conv(radius).to(self.device) for radius in range(1, 8)
        ])

        return seg_model, cls_model


    def initSegmentationDl(self, series_uid):
        seg_ds = Luna2dSegmentationDataset(
                contextSlices_count=3,
                series_uid=series_uid,
                fullCt_bool=True,
            )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=1, #self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl

    def initClassificationDl(self, candidateInfo_list):
        cls_ds = LunaDataset(
                sortby_str='series_uid',
                candidateInfo_list=candidateInfo_list,
            )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=1, #self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return cls_dl


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )
        val_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in val_ds.candidateInfo_list
        )
        positive_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in getCandidateInfoList()
            if candidateInfo_tup.isNodule_bool
        )

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(
                candidateInfo_tup.series_uid
                for candidateInfo_tup in getCandidateInfoList()
            )

        train_list = sorted(series_set - val_set) if self.cli_args.include_train else []
        val_list = sorted(series_set & val_set)


        total_tp = total_tn = total_fp = total_fn = 0
        total_missed_pos = 0
        missed_pos_dist_list = []
        missed_pos_cit_list = []
        candidateInfo_dict = getCandidateInfoDict()
        # series2results_dict = {}
        # seg_candidateInfo_list = []
        series_iter = enumerateWithEstimate(
            val_list + train_list,
            "Series",
        )
        for _series_ndx, series_uid in series_iter:
            ct, _output_g, _mask_g, clean_g = self.segmentCt(series_uid)

            seg_candidateInfo_list, _seg_centerIrc_list, _ = self.clusterSegmentationOutput(
                series_uid,
                ct,
                clean_g,
            )
            if not seg_candidateInfo_list:
                continue

            cls_dl = self.initClassificationDl(seg_candidateInfo_list)
            results_list = []

            # batch_iter = enumerateWithEstimate(
            #     cls_dl,
            #     "Cls all",
            #     start_ndx=cls_dl.num_workers,
            # )
            # for batch_ndx, batch_tup in batch_iter:
            for batch_ndx, batch_tup in enumerate(cls_dl):
                input_t, label_t, index_t, series_list, center_t = batch_tup

                input_g = input_t.to(self.device)
                with torch.no_grad():
                    _logits_g, probability_g = self.cls_model(input_g)
                probability_t = probability_g.to('cpu')
                # probability_t = torch.tensor([[0, 1]] * input_t.shape[0], dtype=torch.float32)

                for i, _series_uid in enumerate(series_list):
                    assert series_uid == _series_uid, repr([batch_ndx, i, series_uid, _series_uid, seg_candidateInfo_list])
                    results_list.append((center_t[i], probability_t[i,0].item()))



            # This part is all about matching up annotations with our segmentation results
            tp = tn = fp = fn = 0
            missed_pos = 0
            ct = getCt(series_uid)
            candidateInfo_list = candidateInfo_dict[series_uid]
            candidateInfo_list = [cit for cit in candidateInfo_list if cit.isNodule_bool]

            found_cit_list = [None] * len(results_list)

            for candidateInfo_tup in candidateInfo_list:
                min_dist = (999, None)

                for result_ndx, (result_center_irc_t, nodule_probability_t) in enumerate(results_list):
                    result_center_xyz = irc2xyz(result_center_irc_t, ct.origin_xyz, ct.vxSize_xyz, ct.direction_a)
                    delta_xyz_t = torch.tensor(result_center_xyz) - torch.tensor(candidateInfo_tup.center_xyz)
                    distance_t = (delta_xyz_t ** 2).sum().sqrt()

                    min_dist = min(min_dist, (distance_t, result_ndx))

                distance_cutoff = max(10, candidateInfo_tup.diameter_mm / 2)
                if min_dist[0] < distance_cutoff:
                    found_dist, result_ndx = min_dist
                    nodule_probability_t = results_list[result_ndx][1]

                    assert candidateInfo_tup.isNodule_bool

                    if nodule_probability_t > 0.5:
                        tp += 1
                    else:
                        fn += 1

                    found_cit_list[result_ndx] = candidateInfo_tup

                else:
                    log.warning("!!! Missed positive {}; {} min dist !!!".format(candidateInfo_tup, min_dist))
                    missed_pos += 1
                    missed_pos_dist_list.append(float(min_dist[0]))
                    missed_pos_cit_list.append(candidateInfo_tup)

            # # TODO remove
            # acceptable_set = {
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860',
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249',
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.195557219224169985110295082004',
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.216252660192313507027754194207',
            #     # '1.3.6.1.4.1.14519.5.2.1.6279.6001.229096941293122177107846044795',
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.229096941293122177107846044795',
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.299806338046301317870803017534',
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.395623571499047043765181005112',
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.487745546557477250336016826588',
            #     '1.3.6.1.4.1.14519.5.2.1.6279.6001.970428941353693253759289796610',
            # }
            # if missed_pos > 0 and series_uid not in acceptable_set:
            #     log.info("Unacceptable series_uid: " + series_uid)
            #     break
            #
            # if total_missed_pos > 10:
            #     break
            #
            #
            # for result_ndx, (result_center_irc_t, nodule_probability_t) in enumerate(results_list):
            #     if found_cit_list[result_ndx] is None:
            #         if nodule_probability_t > 0.5:
            #             fp += 1
            #         else:
            #             tn += 1


            log.info("{}: {} missed pos, {} fn, {} fp, {} tp, {} tn".format(series_uid, missed_pos, fn, fp, tp, tn))
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_missed_pos += missed_pos

        with open(self.cli_args.segmentation_path, 'rb') as f:
            log.info(self.cli_args.segmentation_path)
            log.info(hashlib.sha1(f.read()).hexdigest())
        with open(self.cli_args.classification_path, 'rb') as f:
            log.info(self.cli_args.classification_path)
            log.info(hashlib.sha1(f.read()).hexdigest())
        log.info("{}: {} missed pos, {} fn, {} fp, {} tp, {} tn".format('total', total_missed_pos, total_fn, total_fp, total_tp, total_tn))
        # missed_pos_dist_list.sort()
        # log.info("missed_pos_dist_list {}".format(missed_pos_dist_list))
        for cit, dist in zip(missed_pos_cit_list, missed_pos_dist_list):
            log.info("    Missed by {}: {}".format(dist, cit))


    def segmentCt(self, series_uid):
        with torch.no_grad():
            ct = getCt(series_uid)

            output_g = torch.zeros(ct.hu_a.shape, dtype=torch.float32, device=self.device)

            seg_dl = self.initSegmentationDl(series_uid)
            for batch_tup in seg_dl:
                input_t, label_t, series_list, slice_ndx_list = batch_tup

                input_g = input_t.to(self.device)
                prediction_g = self.seg_model(input_g)

                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_g[slice_ndx] = prediction_g[i,0]

            mask_g = output_g > 0.5
            clean_g = self.erode(mask_g.unsqueeze(0).unsqueeze(0), 1)[0][0]

            # mask_a = output_a > 0.5
            # clean_a = morph.binary_erosion(mask_a, iterations=1)
            # clean_a = morph.binary_dilation(clean_a, iterations=2)

        return ct, output_g, mask_g, clean_g

    def _make_circle_conv(self, radius):
        diameter = 1 + radius * 2

        a = torch.linspace(-1, 1, steps=diameter)**2
        b = (a[None] + a[:, None])**0.5

        circle_weights = (b <= 1.0).to(torch.float32)

        conv = nn.Conv3d(1, 1, kernel_size=(1, diameter, diameter), padding=(0, radius, radius), bias=False)
        conv.weight.data.fill_(1)
        conv.weight.data *= circle_weights / circle_weights.sum()

        return conv

    def erode(self, input_mask, radius, threshold=1):
        conv = self.conv_list[radius - 1]
        input_float = input_mask.to(torch.float32)
        result = conv(input_float)

        # log.debug(['erode in ', radius, threshold, input_float.min().item(), input_float.mean().item(), input_float.max().item()])
        # log.debug(['erode out', radius, threshold, result.min().item(), result.mean().item(), result.max().item()])

        return result >= threshold


    def clusterSegmentationOutput(self, series_uid,  ct, clean_g):
        clean_a = clean_g.cpu().numpy()
        candidateLabel_a, candidate_count = measure.label(clean_a)
        centerIrc_list = measure.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,
            labels=candidateLabel_a,
            index=list(range(1, candidate_count+1)),
        )

        candidateInfo_list = []
        for i, center_irc in enumerate(centerIrc_list):
            assert np.isfinite(center_irc).all(), repr([series_uid, i, candidate_count, (ct.hu_a[candidateLabel_a == i+1]).sum(), center_irc])
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )
            diameter_mm = 0.0
            # pixel_count = (candidateLabel_a == i+1).sum()
            # area_mm2 = pixel_count * ct.vxSize_xyz[0] * ct.vxSize_xyz[1]
            # diameter_mm = 2 * (area_mm2 / math.pi) ** 0.5

            candidateInfo_tup = \
                CandidateInfoTuple(None, None, None, diameter_mm, series_uid, center_xyz)
            candidateInfo_list.append(candidateInfo_tup)

        return candidateInfo_list, centerIrc_list, candidateLabel_a

    # def logResults(self, mode_str, filtered_list, series2diagnosis_dict, positive_set):
    #     count_dict = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    #     for series_uid in filtered_list:
    #         probablity_float, center_irc = series2diagnosis_dict.get(series_uid, (0.0, None))
    #         if center_irc is not None:
    #             center_irc = tuple(int(x.item()) for x in center_irc)
    #         positive_bool = series_uid in positive_set
    #         prediction_bool = probablity_float > 0.5
    #         correct_bool = positive_bool == prediction_bool
    #
    #         if positive_bool and prediction_bool:
    #             count_dict['tp'] += 1
    #         if not positive_bool and not prediction_bool:
    #             count_dict['tn'] += 1
    #         if not positive_bool and prediction_bool:
    #             count_dict['fp'] += 1
    #         if positive_bool and not prediction_bool:
    #             count_dict['fn'] += 1
    #
    #
    #         log.info("{} {} Label:{!r:5} Pred:{!r:5} Correct?:{!r:5} Value:{:.4f} {}".format(
    #             mode_str,
    #             series_uid,
    #             positive_bool,
    #             prediction_bool,
    #             correct_bool,
    #             probablity_float,
    #             center_irc,
    #         ))
    #
    #     total_count = sum(count_dict.values())
    #     percent_dict = {k: v / (total_count or 1) * 100 for k, v in count_dict.items()}
    #
    #     precision = percent_dict['p'] = count_dict['tp'] / ((count_dict['tp'] + count_dict['fp']) or 1)
    #     recall    = percent_dict['r'] = count_dict['tp'] / ((count_dict['tp'] + count_dict['fn']) or 1)
    #     percent_dict['f1'] = 2 * (precision * recall) / ((precision + recall) or 1)
    #
    #     log.info(mode_str + " tp:{tp:.1f}%, tn:{tn:.1f}%, fp:{fp:.1f}%, fn:{fn:.1f}%".format(
    #         **percent_dict,
    #     ))
    #     log.info(mode_str + " precision:{p:.3f}, recall:{r:.3f}, F1:{f1:.3f}".format(
    #         **percent_dict,
    #     ))



if __name__ == '__main__':
    FalsePosRateCheckApp().main()
