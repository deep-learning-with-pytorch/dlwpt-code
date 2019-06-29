import argparse
import datetime
import os
import socket
import sys

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from util.logconf import logging
from util.util import xyz2irc
from .model import UNetWrapper

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_tensor/metrics_ary
METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
METRICS_MAL_LOSS_NDX = 2
METRICS_BEN_LOSS_NDX = 3

METRICS_MTP_NDX = 4
METRICS_MFN_NDX = 5
METRICS_MFP_NDX = 6
METRICS_BTP_NDX = 7
METRICS_BFN_NDX = 8
# METRICS_BFP_NDX = 9

METRICS_SIZE = 9

class LunaTrainingApp(object):
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=24,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )
        # parser.add_argument('--augment-offset',
        #     help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
        #     action='store_true',
        #     default=False,
        # )
        # parser.add_argument('--augment-scale',
        #     help="Augment the training data by randomly increasing or decreasing the size of the nodule.",
        #     action='store_true',
        #     default=False,
        # )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--tb-prefix',
            default='p2ch12',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='none',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.tst_writer = None

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # # TODO: remove this if block before print
        # # This is due to an odd setup that the author is using to test the code; please ignore for now
        # if socket.gethostname() == 'c2':
        #     self.device = torch.device("cuda:1")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.totalTrainingSamples_count = 0

        augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            augmentation_dict['noise'] = 25.0
        self.augmentation_dict = augmentation_dict


    def initModel(self):
        # model = UNetWrapper(in_channels=8, n_classes=2, depth=3, wf=6, padding=True, batch_norm=True, up_mode='upconv')
        model = UNetWrapper(in_channels=7, n_classes=1, depth=4, wf=3, padding=True, batch_norm=True, up_mode='upconv')

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.01, momentum=0.99)
        # return Adam(self.model.parameters())


    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            test_stride=10,
            contextSlices_count=3,
            augmentation_dict=self.augmentation_dict,
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initTestDl(self):
        test_ds = Luna2dSegmentationDataset(
            test_stride=10,
            contextSlices_count=3,
        )

        test_dl = DataLoader(
            test_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return test_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_seg_' + self.cli_args.comment)
            self.tst_writer = SummaryWriter(log_dir=log_dir + '_tst_seg_' + self.cli_args.comment)


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        test_dl = self.initTestDl()

        self.initTensorboardWriters()
        # self.logModelMetrics(self.model)

        best_score = 0.0

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(test_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trainingMetrics_tensor = self.doTraining(epoch_ndx, train_dl)
            self.logPerformanceMetrics(epoch_ndx, 'trn', trainingMetrics_tensor)
            self.logImages(epoch_ndx, train_dl, test_dl)
            # self.logModelMetrics(self.model)

            testingMetrics_tensor = self.doTesting(epoch_ndx, test_dl)
            score = self.logPerformanceMetrics(epoch_ndx, 'tst', testingMetrics_tensor)
            best_score = max(score, best_score)

            self.saveModel('seg' if self.cli_args.segmentation else 'cls', epoch_ndx, score == best_score)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.tst_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        trainingMetrics_devtensor = torch.zeros(METRICS_SIZE, len(train_dl.dataset)).to(self.device)
        self.model.train()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trainingMetrics_devtensor)
            loss_var.backward()

            self.optimizer.step()
            del loss_var

        self.totalTrainingSamples_count += trainingMetrics_devtensor.size(1)

        return trainingMetrics_devtensor.to('cpu')

    def doTesting(self, epoch_ndx, test_dl):
        with torch.no_grad():
            testingMetrics_devtensor = torch.zeros(METRICS_SIZE, len(test_dl.dataset)).to(self.device)
            self.model.eval()

            batch_iter = enumerateWithEstimate(
                test_dl,
                "E{} Testing ".format(epoch_ndx),
                start_ndx=test_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, test_dl.batch_size, testingMetrics_devtensor)

        return testingMetrics_devtensor.to('cpu')


    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_devtensor):
        input_tensor, label_tensor, label_list, ben_tensor, mal_tensor, _series_list, _start_list = batch_tup

        input_devtensor = input_tensor.to(self.device, non_blocking=True)
        label_devtensor = label_tensor.to(self.device, non_blocking=True)
        mal_devtensor = mal_tensor.to(self.device, non_blocking=True)
        ben_devtensor = ben_tensor.to(self.device, non_blocking=True)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_tensor.size(0)
        intersectionSum = lambda a, b: (a * b).view(a.size(0), -1).sum(dim=1)

        prediction_devtensor = self.model(input_devtensor)
        diceLoss_devtensor = self.diceLoss(label_devtensor, prediction_devtensor)

        with torch.no_grad():
            predictionBool_devtensor = (prediction_devtensor > 0.5).to(torch.float32)

            metrics_devtensor[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_list
            metrics_devtensor[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_devtensor

            malPred_devtensor = predictionBool_devtensor * (1 - ben_devtensor)

            tp = intersectionSum(    mal_devtensor,     malPred_devtensor)
            fn = intersectionSum(    mal_devtensor, 1 - malPred_devtensor)
            fp = intersectionSum(1 - mal_devtensor,     malPred_devtensor)
            ls = self.diceLoss(mal_devtensor, malPred_devtensor)

            metrics_devtensor[METRICS_MTP_NDX, start_ndx:end_ndx] = tp
            metrics_devtensor[METRICS_MFN_NDX, start_ndx:end_ndx] = fn
            metrics_devtensor[METRICS_MFP_NDX, start_ndx:end_ndx] = fp
            metrics_devtensor[METRICS_MAL_LOSS_NDX, start_ndx:end_ndx] = ls

            del malPred_devtensor, tp, fn, fp, ls

            benPred_devtensor = predictionBool_devtensor * (1 - mal_devtensor)
            tp = intersectionSum(    ben_devtensor,     benPred_devtensor)
            fn = intersectionSum(    ben_devtensor, 1 - benPred_devtensor)
            # fp = intersectionSum(1 - ben_devtensor,     benPred_devtensor)
            ls = self.diceLoss(ben_devtensor, benPred_devtensor)

            metrics_devtensor[METRICS_BTP_NDX, start_ndx:end_ndx] = tp
            metrics_devtensor[METRICS_BFN_NDX, start_ndx:end_ndx] = fn
            # metrics_devtensor[METRICS_BFP_NDX, start_ndx:end_ndx] = fp
            metrics_devtensor[METRICS_BEN_LOSS_NDX, start_ndx:end_ndx] = ls

            del benPred_devtensor, tp, fn, ls

        return diceLoss_devtensor.mean()

    def diceLoss(self, label_devtensor, prediction_devtensor, epsilon=0.01, p=False):
        sum_dim1 = lambda t: t.view(t.size(0), -1).sum(dim=1)

        diceLabel_devtensor = sum_dim1(label_devtensor)
        dicePrediction_devtensor = sum_dim1(prediction_devtensor)
        diceCorrect_devtensor = sum_dim1(prediction_devtensor * label_devtensor)

        epsilon_devtensor = torch.ones_like(diceCorrect_devtensor) * epsilon
        diceLoss_devtensor = 1 - (2 * diceCorrect_devtensor + epsilon_devtensor) / (dicePrediction_devtensor + diceLabel_devtensor + epsilon_devtensor)

        return diceLoss_devtensor



    def logImages(self, epoch_ndx, train_dl, test_dl):
        for mode_str, dl in [('trn', train_dl), ('tst', test_dl)]:
            for i, series_uid in enumerate(sorted(dl.dataset.series_list)[:12]):
                ct = getCt(series_uid)
                noduleInfo_tup = (ct.malignantInfo_list or ct.benignInfo_list)[0]
                center_irc = xyz2irc(noduleInfo_tup.center_xyz, ct.origin_xyz, ct.vxSize_xyz, ct.direction_tup)

                sample_tup = dl.dataset[(series_uid, int(center_irc.index))]
                # input_tensor = sample_tup[0].unsqueeze(0)
                # label_tensor = sample_tup[1].unsqueeze(0)

                input_tensor, label_tensor, ben_tensor, mal_tensor = sample_tup[:4]
                input_tensor += 1000
                input_tensor /= 2001

                input_devtensor = input_tensor.to(self.device)
                # label_devtensor = label_tensor.to(self.device)

                prediction_devtensor = self.model(input_devtensor.unsqueeze(0))[0]
                prediction_ary = prediction_devtensor.to('cpu').detach().numpy()
                label_ary = label_tensor.numpy()
                ben_ary = ben_tensor.numpy()
                mal_ary = mal_tensor.numpy()

                # log.debug([prediction_ary[0].shape, label_ary.shape, mal_ary.shape])

                image_ary = np.zeros((512, 512, 3), dtype=np.float32)
                image_ary[:,:,:] = (input_tensor[dl.dataset.contextSlices_count].numpy().reshape((512,512,1))) * 0.5
                image_ary[:,:,0] += prediction_ary[0] * (1 - label_ary[0]) * 0.5
                image_ary[:,:,1] += prediction_ary[0] * mal_ary * 0.5
                image_ary[:,:,2] += prediction_ary[0] * ben_ary * 0.5
                # image_ary[:,:,2] += prediction_ary[0,1] * 0.25
                # image_ary[:,:,2] += prediction_ary[0,2] * 0.5

                # log.debug([image_ary.__array_interface__['typestr']])

                # image_ary = (image_ary * 255).astype(np.uint8)

                # log.debug([image_ary.__array_interface__['typestr']])

                writer = getattr(self, mode_str + '_writer')
                try:
                    image_ary[image_ary < 0] = 0
                    image_ary[image_ary > 1] = 1
                    writer.add_image('{}/{}_pred'.format(mode_str, i), image_ary, self.totalTrainingSamples_count, dataformats='HWC')
                except:
                    log.debug([image_ary.shape, image_ary.dtype])
                    raise

                if epoch_ndx == 1:

                    image_ary = np.zeros((512, 512, 3), dtype=np.float32)
                    image_ary[:,:,:] = (input_tensor[dl.dataset.contextSlices_count].numpy().reshape((512,512,1))) * 0.5
                    image_ary[:,:,1] += mal_ary * 0.5
                    image_ary[:,:,2] += ben_ary * 0.5
                    # image_ary[:,:,2] += label_ary[0,1] * 0.25
                    # image_ary[:,:,2] += (input_tensor[0,-1].numpy() - (label_ary[0,0].astype(np.bool) | label_ary[0,1].astype(np.bool))) * 0.25

                    # log.debug([image_ary.__array_interface__['typestr']])

                    # image_ary = (image_ary * 255).astype(np.uint8)

                    # log.debug([image_ary.__array_interface__['typestr']])

                    writer = getattr(self, mode_str + '_writer')
                    image_ary[image_ary < 0] = 0
                    image_ary[image_ary > 1] = 1
                    writer.add_image('{}/{}_label'.format(mode_str, i), image_ary, self.totalTrainingSamples_count, dataformats='HWC')


    def logPerformanceMetrics(self,
                              epoch_ndx,
                              mode_str,
                              metrics_tensor,
                              ):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))


        # for mode_str, metrics_tensor in [('trn', trainingMetrics_tensor), ('tst', testingMetrics_tensor)]:
        metrics_ary = metrics_tensor.cpu().detach().numpy()
        sum_ary = metrics_ary.sum(axis=1)
        assert np.isfinite(metrics_ary).all()

        malLabel_mask = (metrics_ary[METRICS_LABEL_NDX] == 1) | (metrics_ary[METRICS_LABEL_NDX] == 3)

        benLabel_mask = (metrics_ary[METRICS_LABEL_NDX] == 2) | (metrics_ary[METRICS_LABEL_NDX] == 3)
        # malFound_mask = metrics_ary[METRICS_MFOUND_NDX] > classificationThreshold_float

        # malLabel_mask = ~benLabel_mask
        # malPred_mask = ~benPred_mask

        benLabel_count = sum_ary[METRICS_BTP_NDX] + sum_ary[METRICS_BFN_NDX]
        malLabel_count = sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFN_NDX]

        trueNeg_count = benCorrect_count = sum_ary[METRICS_BTP_NDX]
        truePos_count = malCorrect_count = sum_ary[METRICS_MTP_NDX]
#
#             falsePos_count = benLabel_count - benCorrect_count
#             falseNeg_count = malLabel_count - malCorrect_count


        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_ary[METRICS_LOSS_NDX].mean()
        # metrics_dict['loss/msk'] = metrics_ary[METRICS_MASKLOSS_NDX].mean()
        # metrics_dict['loss/mal'] = metrics_ary[METRICS_MALLOSS_NDX].mean()
        # metrics_dict['loss/lng'] = metrics_ary[METRICS_LUNG_LOSS_NDX, benLabel_mask].mean()
        metrics_dict['loss/mal'] = np.nan_to_num(metrics_ary[METRICS_MAL_LOSS_NDX, malLabel_mask].mean())
        metrics_dict['loss/ben'] = metrics_ary[METRICS_BEN_LOSS_NDX, benLabel_mask].mean()
        # metrics_dict['loss/flg'] = metrics_ary[METRICS_FLG_LOSS_NDX].mean()

        # metrics_dict['flagged/all'] = sum_ary[METRICS_MOK_NDX] / (sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFN_NDX]) * 100
        # metrics_dict['flagged/slices'] = (malLabel_mask & malFound_mask).sum() / malLabel_mask.sum() * 100

        metrics_dict['correct/mal'] = sum_ary[METRICS_MTP_NDX] / (sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFN_NDX]) * 100
        metrics_dict['correct/ben'] = sum_ary[METRICS_BTP_NDX] / (sum_ary[METRICS_BTP_NDX] + sum_ary[METRICS_BFN_NDX]) * 100

        precision = metrics_dict['pr/precision'] = sum_ary[METRICS_MTP_NDX] / ((sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_ary[METRICS_MTP_NDX] / ((sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 # + "{loss/flg:.4f} flagged loss, "
                 # + "{flagged/all:-5.1f}% pixels flagged, "
                 # + "{flagged/slices:-5.1f}% slices flagged, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                 + "{loss/mal:.4f} loss, "
                 + "{correct/mal:-5.1f}% correct ({malCorrect_count:} of {malLabel_count:})"
        ).format(
            epoch_ndx,
            mode_str + '_mal',
            malCorrect_count=malCorrect_count,
            malLabel_count=malLabel_count,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                 + "{loss/ben:.4f} loss, "
                 + "{correct/ben:-5.1f}% correct ({benCorrect_count:} of {benLabel_count:})"
        ).format(
            epoch_ndx,
            mode_str + '_ben',
            benCorrect_count=benCorrect_count,
            benLabel_count=benLabel_count,
            **metrics_dict,
        ))

        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        score = 1 \
            + metrics_dict['pr/f1_score'] \
            - metrics_dict['loss/mal'] * 0.01 \
            - metrics_dict['loss/all'] * 0.0001

        return score

    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             writer.add_histogram(
    #                 name.rsplit('.', 1)[-1] + '/' + name,
    #                 param.data.cpu().numpy(),
    #                 # metrics_ary[METRICS_PRED_NDX, benHist_mask],
    #                 self.totalTrainingSamples_count,
    #                 # bins=bins,
    #             )
    #
    #             # print name, param.data

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join('data-unversioned', 'models', self.cli_args.tb_prefix, '{}_{}_{}.{}.state'.format(type_str, self.time_str, self.cli_args.comment, self.totalTrainingSamples_count))

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if hasattr(model, 'module'):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
            # 'resumed_from': self.cli_args.resume,
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            file_path = os.path.join('data-unversioned', 'models', self.cli_args.tb_prefix, '{}_{}_{}.{}.state'.format(type_str, self.time_str, self.cli_args.comment, 'best'))
            torch.save(state, file_path)

            log.debug("Saved model params to {}".format(file_path))


if __name__ == '__main__':
    sys.exit(LunaTrainingApp().main() or 0)
