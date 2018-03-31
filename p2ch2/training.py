import argparse
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
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

# Used for metrics_ary index 0
LABEL=0
PRED=1
LOSS=2
# ...

class LunaTrainingApp(object):
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=256,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=10,
            type=int,
        )
        parser.add_argument('--layers',
            help='Number of layers to the model',
            default=3,
            type=int,
        )
        parser.add_argument('--channels',
            help="Number of channels for the first layer's convolutions to the model (doubles each layer)",
            default=8,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        self.train_dl = DataLoader(
            LunaDataset(
                test_stride=10,
                isTestSet_bool=False,
            ),
            batch_size=self.cli_args.batch_size * torch.cuda.device_count(),
            num_workers=self.cli_args.num_workers,
            pin_memory=True,
        )
        self.test_dl = DataLoader(
            LunaDataset(
                test_stride=10,
                isTestSet_bool=True,
            ),
            batch_size=self.cli_args.batch_size * torch.cuda.device_count(),
            num_workers=self.cli_args.num_workers,
            pin_memory=True,
        )

        self.model = LunaModel(self.cli_args.layers, 1, self.cli_args.channels)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()

        self.optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(self.train_dl),
                len(self.test_dl),
                self.cli_args.batch_size,
                torch.cuda.device_count(),
            ))
            
            # Training loop, very similar to below
            self.model.train()
            batch_iter = enumerateWithEstimate(
                self.train_dl,
                "E{} Training".format(epoch_ndx),
                start_ndx=self.train_dl.num_workers,
            )
            trainingMetrics_ary = np.zeros((3, len(self.train_dl.dataset)), dtype=np.float32)
            for batch_ndx, batch_tup in batch_iter:
                self.optimizer.zero_grad()
                loss_var = self.computeBatchLoss(batch_ndx, batch_tup, self.train_dl.batch_size, trainingMetrics_ary)
                loss_var.backward()
                self.optimizer.step()
                del loss_var

            # Testing loop, very similar to above, but simplified
            # ...
            self.model.eval()
            batch_iter = enumerateWithEstimate(
                self.test_dl,
                "E{} Testing ".format(epoch_ndx),
                start_ndx=self.test_dl.num_workers,
            )
            testingMetrics_ary = np.zeros((3, len(self.test_dl.dataset)), dtype=np.float32)
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, self.test_dl.batch_size, testingMetrics_ary)

            self.logMetrics(epoch_ndx, trainingMetrics_ary, testingMetrics_ary)


    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_ary):
        input_tensor, label_tensor, series_list, center_list = batch_tup

        input_var = Variable(input_tensor.cuda())
        label_var = Variable(label_tensor.cuda())
        prediction_var = self.model(input_var)
        # ...

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_tensor.size(0)
        metrics_ary[LABEL, start_ndx:end_ndx] = label_tensor.numpy()[:,0,0]
        metrics_ary[PRED,  start_ndx:end_ndx] = prediction_var.data.cpu().numpy()[:,0]

        for sample_ndx in range(label_tensor.size(0)):
            subloss_var = nn.MSELoss()(prediction_var[sample_ndx], label_var[sample_ndx])
            metrics_ary[LOSS, start_ndx+sample_ndx] = subloss_var.data[0]
            del subloss_var

        loss_var = nn.MSELoss()(prediction_var, label_var)
        return loss_var


    def logMetrics(self, epoch_ndx, trainingMetrics_ary, testingMetrics_ary):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        for mode_str, metrics_ary in [('trn', trainingMetrics_ary), ('tst', testingMetrics_ary)]:
            pos_mask = metrics_ary[LABEL] > 0.5
            neg_mask = ~pos_mask

            truePos_count = (metrics_ary[PRED, pos_mask] > 0.5).sum()
            trueNeg_count = (metrics_ary[PRED, neg_mask] < 0.5).sum()

            metrics_dict = {}
            metrics_dict['loss/all'] = metrics_ary[LOSS].mean()
            metrics_dict['loss/ben'] = metrics_ary[LOSS, neg_mask].mean()
            metrics_dict['loss/mal'] = metrics_ary[LOSS, pos_mask].mean()

            metrics_dict['correct/all'] = (truePos_count + trueNeg_count) / metrics_ary.shape[1] * 100
            metrics_dict['correct/ben'] = (trueNeg_count) / neg_mask.sum() * 100
            metrics_dict['correct/mal'] = (truePos_count) / pos_mask.sum() * 100

            log.info("E{} {:8} {loss/all:.4f} loss, {correct/all:-5.1f}% correct".format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            ))
            log.info("E{} {:8} {loss/ben:.4f} loss, {correct/ben:-5.1f}% correct".format(
                epoch_ndx,
                mode_str + '_ben',
                **metrics_dict,
            ))
            log.info("E{} {:8} {loss/mal:.4f} loss, {correct/mal:-5.1f}% correct".format(
                epoch_ndx,
                mode_str + '_mal',
                **metrics_dict,
            ))


if __name__ == '__main__':
    sys.exit(LunaTrainingApp().main() or 0)
