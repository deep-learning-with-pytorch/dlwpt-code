import argparse
import glob
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
from .dsets import LunaDataset, Luna2dSegmentationDataset, getCt, getNoduleInfoList, NoduleInfoTuple
from .model_seg import UNetWrapper
from .model_cls import LunaModel, AlternateLunaModel

from util.logconf import logging
from util.util import xyz2irc, irc2xyz

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class LunaDiagnoseApp(object):
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
        local_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            type_str + '_{}_{}.{}.state'.format('*', '*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                'data',
                'part2',
                'models',
                type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
            )
            file_list = glob.glob(pretrained_path)
        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, pretrained_path, file_list])
            raise

    def initModels(self):
        log.debug(self.cli_args.segmentation_path)
        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = UNetWrapper(in_channels=8, n_classes=1, depth=4, wf=3, padding=True, batch_norm=True, up_mode='upconv')
        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        log.debug(self.cli_args.classification_path)
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
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl

    def initClassificationDl(self, noduleInfo_list):
        cls_ds = LunaDataset(
                sortby_str='series_uid',
                noduleInfo_list=noduleInfo_list,
            )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
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
            noduleInfo_tup.series_uid
            for noduleInfo_tup in val_ds.noduleInfo_list
        )
        malignant_set = set(
            noduleInfo_tup.series_uid
            for noduleInfo_tup in getNoduleInfoList()
            if noduleInfo_tup.isMalignant_bool
        )

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(
                noduleInfo_tup.series_uid
                for noduleInfo_tup in getNoduleInfoList()
            )

        train_list = sorted(series_set - val_set) if self.cli_args.include_train else []
        val_list = sorted(series_set & val_set)


        noduleInfo_list = []
        series_iter = enumerateWithEstimate(
            val_list + train_list,
            "Series",
        )
        for _series_ndx, series_uid in series_iter:
            ct, output_a, _mask_a, clean_a = self.segmentCt(series_uid)

            noduleInfo_list += self.clusterSegmentationOutput(
                series_uid,
                ct,
                clean_a,
            )

            # if _series_ndx > 10:
            #     break


        cls_dl = self.initClassificationDl(noduleInfo_list)

        series2diagnosis_dict = {}
        batch_iter = enumerateWithEstimate(
            cls_dl,
            "Cls all",
            start_ndx=cls_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            input_t, _, series_list, center_list = batch_tup

            input_g = input_t.to(self.device)
            with torch.no_grad():
                _logits_g, probability_g = self.cls_model(input_g)

            classifications_list = zip(
                series_list,
                center_list,
                probability_g[:,1].to('cpu'),
            )

            for cls_tup in classifications_list:
                series_uid, center_irc, probablity_t = cls_tup
                probablity_float = probablity_t.item()

                this_tup = (probablity_float, tuple(center_irc))
                current_tup = series2diagnosis_dict.get(series_uid, this_tup)
                try:
                    assert np.all(np.isfinite(tuple(center_irc)))
                    if this_tup > current_tup:
                        log.debug([series_uid, this_tup])
                    series2diagnosis_dict[series_uid] = max(this_tup, current_tup)
                except:
                    log.debug([(type(x), x) for x in this_tup] + [(type(x), x) for x in current_tup])
                    raise

        log.info('Training set:')
        self.logResults('Training', train_list, series2diagnosis_dict, malignant_set)

        log.info('Validation set:')
        self.logResults('Validation', val_list, series2diagnosis_dict, malignant_set)

    def segmentCt(self, series_uid):
        with torch.no_grad():
            ct = getCt(series_uid)

            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)

            seg_dl = self.initSegmentationDl(series_uid)
            for batch_tup in seg_dl:
                input_t = batch_tup[0]
                ndx_list = batch_tup[6]

                input_g = input_t.to(self.device)
                prediction_g = self.seg_model(input_g)

                for i, sample_ndx in enumerate(ndx_list):
                    output_a[sample_ndx] = prediction_g[i].cpu().numpy()

            mask_a = output_a > 0.5
            clean_a = morph.binary_erosion(mask_a, iterations=1)
            clean_a = morph.binary_dilation(clean_a, iterations=2)

        return ct, output_a, mask_a, clean_a

    def clusterSegmentationOutput(self, series_uid,  ct, clean_a):
        noduleLabel_a, nodule_count = measure.label(clean_a)
        centerIrc_list = measure.center_of_mass(
            ct.hu_a + 1001,
            labels=noduleLabel_a,
            index=list(range(1, nodule_count+1)),
        )

        # n = 1298
        # log.debug([
        #     (noduleLabel_a == n).sum(),
        #     np.where(noduleLabel_a == n),
        #
        #     ct.hu_a[noduleLabel_a == n].sum(),
        #     (ct.hu_a + 1000)[noduleLabel_a == n].sum(),
        # ])

        # if nodule_count == 1:
        #     centerIrc_list = [centerIrc_list]

        noduleInfo_list = []
        for i, center_irc in enumerate(centerIrc_list):
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_tup,
            )
            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, nodule_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
            noduleInfo_tup = \
                NoduleInfoTuple(False, 0.0, series_uid, center_xyz)
            noduleInfo_list.append(noduleInfo_tup)

        return noduleInfo_list

    def logResults(self, mode_str, filtered_list, series2diagnosis_dict, malignant_set):
        count_dict = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        for series_uid in filtered_list:
            probablity_float, center_irc = series2diagnosis_dict.get(series_uid, (0.0, None))
            if center_irc is not None:
                center_irc = tuple(int(x.item()) for x in center_irc)
            malignant_bool = series_uid in malignant_set
            prediction_bool = probablity_float > 0.5
            correct_bool = malignant_bool == prediction_bool

            if malignant_bool and prediction_bool:
                count_dict['tp'] += 1
            if not malignant_bool and not prediction_bool:
                count_dict['tn'] += 1
            if not malignant_bool and prediction_bool:
                count_dict['fp'] += 1
            if malignant_bool and not prediction_bool:
                count_dict['fn'] += 1


            log.info("{} {} Mal:{!r:5} Pred:{!r:5} Correct?:{!r:5} Value:{:.4f} {}".format(
                mode_str,
                series_uid,
                malignant_bool,
                prediction_bool,
                correct_bool,
                probablity_float,
                center_irc,
            ))

        total_count = sum(count_dict.values())
        percent_dict = {k: v / (total_count or 1) * 100 for k, v in count_dict.items()}

        precision = percent_dict['p'] = count_dict['tp'] / ((count_dict['tp'] + count_dict['fp']) or 1)
        recall    = percent_dict['r'] = count_dict['tp'] / ((count_dict['tp'] + count_dict['fn']) or 1)
        percent_dict['f1'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(mode_str + " tp:{tp:.1f}%, tn:{tn:.1f}%, fp:{fp:.1f}%, fn:{fn:.1f}%".format(
            **percent_dict,
        ))
        log.info(mode_str + " precision:{p:.3f}, recall:{r:.3f}, F1:{f1:.3f}".format(
            **percent_dict,
        ))



if __name__ == '__main__':
    sys.exit(LunaDiagnoseApp().main() or 0)
