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

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = LunaModel()
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

            self.model = self.model.to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        train_dl = DataLoader(
            LunaDataset(
                test_stride=10,
                isTestSet_bool=False,
            ),
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        test_dl = DataLoader(
            LunaDataset(
                test_stride=10,
                isTestSet_bool=True,
            ),
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(test_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            # Training loop, very similar to below
            self.model.train()
            trainingMetrics_tensor = torch.zeros(3, len(train_dl.dataset), 1)
            batch_iter = enumerateWithEstimate(
                train_dl,
                "E{} Training".format(epoch_ndx),
                start_ndx=train_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.optimizer.zero_grad()
                loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trainingMetrics_tensor)
                loss_var.backward()
                self.optimizer.step()
                del loss_var

            # Testing loop, very similar to above, but simplified
            with torch.no_grad():
                self.model.eval()
                testingMetrics_tensor = torch.zeros(3, len(test_dl.dataset), 1)
                batch_iter = enumerateWithEstimate(
                    test_dl,
                    "E{} Testing ".format(epoch_ndx),
                    start_ndx=test_dl.num_workers,
                )
                for batch_ndx, batch_tup in batch_iter:
                    self.computeBatchLoss(batch_ndx, batch_tup, test_dl.batch_size, testingMetrics_tensor)

            self.logMetrics(epoch_ndx, trainingMetrics_tensor, testingMetrics_tensor)


    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_tensor):
        input_tensor, label_tensor, _series_list, _center_list = batch_tup

        input_devtensor = input_tensor.to(self.device)
        label_devtensor = label_tensor.to(self.device)

        prediction_devtensor = self.model(input_devtensor)
        loss_devtensor = nn.MSELoss(reduction='none')(prediction_devtensor, label_devtensor)


        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_tensor.size(0)
        metrics_tensor[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_tensor
        metrics_tensor[METRICS_PRED_NDX, start_ndx:end_ndx] = prediction_devtensor.to('cpu')
        metrics_tensor[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_devtensor.to('cpu')

        # TODO: replace with torch.autograd.detect_anomaly
        # assert np.isfinite(metrics_tensor).all()

        return loss_devtensor.mean()


    def logMetrics(self,
                   epoch_ndx,
                   trainingMetrics_tensor,
                   testingMetrics_tensor,
                   classificationThreshold_float=0.5,
                   ):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        for mode_str, metrics_tensor in [('trn', trainingMetrics_tensor), ('tst', testingMetrics_tensor)]:
            metrics_ary = metrics_tensor.detach().numpy()[:,:,0]
            assert np.isfinite(metrics_ary).all()

            benLabel_mask = metrics_ary[METRICS_LABEL_NDX] <= classificationThreshold_float
            benPred_mask = metrics_ary[METRICS_PRED_NDX] <= classificationThreshold_float

            malLabel_mask = ~benLabel_mask
            malPred_mask = ~benPred_mask

            benLabel_count = benLabel_mask.sum()
            malLabel_count = malLabel_mask.sum()

            benCorrect_count = (benLabel_mask & benPred_mask).sum()
            malCorrect_count = (malLabel_mask & malPred_mask).sum()

            metrics_dict = {}

            metrics_dict['loss/all'] = metrics_ary[METRICS_LOSS_NDX].mean()
            metrics_dict['loss/ben'] = metrics_ary[METRICS_LOSS_NDX, benLabel_mask].mean()
            metrics_dict['loss/mal'] = metrics_ary[METRICS_LOSS_NDX, malLabel_mask].mean()

            metrics_dict['correct/all'] = (malCorrect_count + benCorrect_count) / metrics_ary.shape[1] * 100
            metrics_dict['correct/ben'] = (benCorrect_count) / benLabel_count * 100
            metrics_dict['correct/mal'] = (malCorrect_count) / malLabel_count * 100




            log.info(("E{} {:8} "
                     + "{loss/all:.4f} loss, "
                     + "{correct/all:-5.1f}% correct"
                      ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            ))
            log.info(("E{} {:8} "
                     + "{loss/ben:.4f} loss, "
                     + "{correct/ben:-5.1f}% correct").format(
                epoch_ndx,
                mode_str + '_ben',
                **metrics_dict,
            ))
            log.info(("E{} {:8} "
                     + "{loss/mal:.4f} loss, "
                     + "{correct/mal:-5.1f}% correct").format(
                epoch_ndx,
                mode_str + '_mal',
                **metrics_dict,
            ))


if __name__ == '__main__':
    sys.exit(LunaTrainingApp().main() or 0)
