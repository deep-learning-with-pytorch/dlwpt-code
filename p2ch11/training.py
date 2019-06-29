import argparse
import datetime
import os
import sys

import numpy as np

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import LunaDataset
from util.logconf import logging
from .model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_tensor/metrics_ary
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

class LunaTrainingApp(object):
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
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
        parser.add_argument('--balanced',
            help="Balance the training data to half benign, half malignant.",
            action='store_true',
            default=False,
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
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the nodule.",
            action='store_true',
            default=False,
        )
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
            default='p2ch11',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )
        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='none',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.tst_writer = None

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()


    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())

    def initTrainDl(self):
        train_ds = LunaDataset(
            test_stride=10,
            isTestSet_bool=False,
            ratio_int=int(self.cli_args.balanced),
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
        test_ds = LunaDataset(
            test_stride=10,
            isTestSet_bool=True,
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

            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_cls_' + self.cli_args.comment)
            self.tst_writer = SummaryWriter(log_dir=log_dir + '_tst_cls_' + self.cli_args.comment)
# eng::tb_writer[]


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        test_dl = self.initTestDl()

        self.initTensorboardWriters()
        # self.logModelMetrics(self.model)

        # best_score = 0.0

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(test_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_tensor = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_tensor)

            tstMetrics_tensor = self.doTesting(epoch_ndx, test_dl)
            self.logMetrics(epoch_ndx, 'tst', tstMetrics_tensor)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.tst_writer.close()


    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trainingMetrics_devtensor = torch.zeros(METRICS_SIZE, len(train_dl.dataset)).to(self.device)
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trainingMetrics_devtensor
            )

            loss_var.backward()
            self.optimizer.step()
            del loss_var

        self.totalTrainingSamples_count += trainingMetrics_devtensor.size(1)

        return trainingMetrics_devtensor.to('cpu')


    def doTesting(self, epoch_ndx, test_dl):
        with torch.no_grad():
            self.model.eval()
            testingMetrics_devtensor = torch.zeros(METRICS_SIZE, len(test_dl.dataset)).to(self.device)
            batch_iter = enumerateWithEstimate(
                test_dl,
                "E{} Testing ".format(epoch_ndx),
                start_ndx=test_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, test_dl.batch_size, testingMetrics_devtensor)

        return testingMetrics_devtensor.to('cpu')



    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_devtensor):
        input_tensor, label_tensor, _series_list, _center_list = batch_tup

        input_devtensor = input_tensor.to(self.device, non_blocking=True)
        label_devtensor = label_tensor.to(self.device, non_blocking=True)

        logits_devtensor, probability_devtensor = self.model(input_devtensor)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_devtensor = loss_func(logits_devtensor, label_devtensor[:,1])
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_tensor.size(0)

        metrics_devtensor[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_devtensor[:,1]
        metrics_devtensor[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_devtensor[:,1]
        metrics_devtensor[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_devtensor

        return loss_devtensor.mean()


    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_tensor,
    ):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_ary = metrics_tensor.cpu().detach().numpy()
#         assert np.isfinite(metrics_ary).all()

        benLabel_mask = metrics_ary[METRICS_LABEL_NDX] <= 0.5
        benPred_mask = metrics_ary[METRICS_PRED_NDX] <= 0.5

        malLabel_mask = ~benLabel_mask
        malPred_mask = ~benPred_mask

        benLabel_count = benLabel_mask.sum()
        malLabel_count = malLabel_mask.sum()

        trueNeg_count = benCorrect_count = (benLabel_mask & benPred_mask).sum()
        truePos_count = malCorrect_count = (malLabel_mask & malPred_mask).sum()

        falsePos_count = benLabel_count - benCorrect_count
        falseNeg_count = malLabel_count - malCorrect_count

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_ary[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/ben'] = metrics_ary[METRICS_LOSS_NDX, benLabel_mask].mean()
        metrics_dict['loss/mal'] = metrics_ary[METRICS_LOSS_NDX, malLabel_mask].mean()

        metrics_dict['correct/all'] = (malCorrect_count + benCorrect_count) / metrics_ary.shape[1] * 100
        metrics_dict['correct/ben'] = (benCorrect_count) / benLabel_count * 100
        metrics_dict['correct/mal'] = (malCorrect_count) / malLabel_count * 100

        precision = metrics_dict['pr/precision'] = truePos_count / (truePos_count + falsePos_count)
        recall    = metrics_dict['pr/recall']    = truePos_count / (truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        log.info(
            ("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} "
                 + "{loss/ben:.4f} loss, "
                 + "{correct/ben:-5.1f}% correct ({benCorrect_count:} of {benLabel_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_ben',
                benCorrect_count=benCorrect_count,
                benLabel_count=benLabel_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} "
                 + "{loss/mal:.4f} loss, "
                 + "{correct/mal:-5.1f}% correct ({malCorrect_count:} of {malLabel_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_mal',
                malCorrect_count=malCorrect_count,
                malLabel_count=malLabel_count,
                **metrics_dict,
            )
        )
        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_ary[METRICS_LABEL_NDX],
            metrics_ary[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x/50.0 for x in range(51)]

        benHist_mask = benLabel_mask & (metrics_ary[METRICS_PRED_NDX] > 0.01)
        malHist_mask = malLabel_mask & (metrics_ary[METRICS_PRED_NDX] < 0.99)

        if benHist_mask.any():
            writer.add_histogram(
                'is_ben',
                metrics_ary[METRICS_PRED_NDX, benHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if malHist_mask.any():
            writer.add_histogram(
                'is_mal',
                metrics_ary[METRICS_PRED_NDX, malHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        # score = 1 \
        #     + metrics_dict['pr/f1_score'] \
        #     - metrics_dict['loss/mal'] * 0.01 \
        #     - metrics_dict['loss/all'] * 0.0001
        #
        # return score

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
    #             try:
    #                 writer.add_histogram(
    #                     name.rsplit('.', 1)[-1] + '/' + name,
    #                     param.data.cpu().numpy(),
    #                     # metrics_ary[METRICS_PRED_NDX, benHist_mask],
    #                     self.totalTrainingSamples_count,
    #                     # bins=bins,
    #                 )
    #             except Exception as e:
    #                 log.error([min_data, max_data])
    #                 raise


if __name__ == '__main__':
    sys.exit(LunaTrainingApp().main() or 0)
