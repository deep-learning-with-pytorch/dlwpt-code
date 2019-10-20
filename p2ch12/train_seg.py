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
from .model_seg import UNetWrapper

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
METRICS_MAL_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

METRICS_MTP_NDX = 4
METRICS_MFN_NDX = 5
METRICS_MFP_NDX = 6
METRICS_ATP_NDX = 7
METRICS_AFN_NDX = 8
METRICS_AFP_NDX = 9

METRICS_SIZE = 10

class LunaTrainingApp(object):
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=16,
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
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None

        augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            augmentation_dict['noise'] = 0.025
        self.augmentation_dict = augmentation_dict

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()


    def initModel(self):
        model = UNetWrapper(
            in_channels=8,
            n_classes=1,
            depth=4,
            wf=3,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())


    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
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

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_seg_' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '_val_seg_' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        # self.logModelMetrics(self.model)

        best_score = 0.0
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)
            self.logImages(epoch_ndx, 'trn', train_dl)
            self.logImages(epoch_ndx, 'val', val_dl)
            # self.logModelMetrics(self.model)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            best_score = max(score, best_score)

            self.saveModel('seg', epoch_ndx, score == best_score)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset)).to(self.device)
        self.model.train()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()

            self.optimizer.step()
            del loss_var

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset)).to(self.device)
            self.model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, label_list, ben_t, mal_t, _, _ = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        mal_g = mal_t.to(self.device, non_blocking=True)
        ben_g = ben_t.to(self.device, non_blocking=True)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        intersectionSum = lambda a, b: (a * b).view(a.size(0), -1).sum(dim=1)

        prediction_g = self.model(input_g)
        diceLoss_g = self.diceLoss(label_g, prediction_g)

        with torch.no_grad():
            malLoss_g = self.diceLoss(mal_g, prediction_g * mal_g, p=True)
            predictionBool_g = (prediction_g > 0.5).to(torch.float32)

            metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_list
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_MAL_LOSS_NDX, start_ndx:end_ndx] = malLoss_g

            malPred_g = predictionBool_g * mal_g
            tp = intersectionSum(    mal_g,       malPred_g)
            fn = intersectionSum(    mal_g,   1 - malPred_g)

            metrics_g[METRICS_MTP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_MFN_NDX, start_ndx:end_ndx] = fn

            del malPred_g, tp, fn

            tp = intersectionSum(    label_g,     predictionBool_g)
            fn = intersectionSum(    label_g, 1 - predictionBool_g)
            fp = intersectionSum(1 - label_g,     predictionBool_g)

            metrics_g[METRICS_ATP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_AFN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_AFP_NDX, start_ndx:end_ndx] = fp

            del tp, fn, fp

        return diceLoss_g.mean()

    # def diceLoss(self, label_g, prediction_g, epsilon=0.01, p=False):
    def diceLoss(self, label_g, prediction_g, epsilon=1, p=False):
        sum_dim1 = lambda t: t.view(t.size(0), -1).sum(dim=1)

        diceLabel_g = sum_dim1(label_g)
        dicePrediction_g = sum_dim1(prediction_g)
        diceCorrect_g = sum_dim1(prediction_g * label_g)

        epsilon_g = torch.ones_like(diceCorrect_g) * epsilon
        diceLoss_g = 1 - (2 * diceCorrect_g + epsilon_g) \
            / (dicePrediction_g + diceLabel_g + epsilon_g)

        if p and diceLoss_g.mean() < 0:
            correct_tmp = prediction_g * label_g

            log.debug([])
            log.debug(['diceCorrect_g   ', diceCorrect_g[0].item(), correct_tmp[0].min().item(), correct_tmp[0].mean().item(), correct_tmp[0].max().item(), correct_tmp.shape])
            log.debug(['dicePrediction_g', dicePrediction_g[0].item(), prediction_g[0].min().item(), prediction_g[0].mean().item(), prediction_g[0].max().item(), prediction_g.shape])
            log.debug(['diceLabel_g     ', diceLabel_g[0].item(), label_g[0].min().item(), label_g[0].mean().item(), label_g[0].max().item(), label_g.shape])
            log.debug(['2*diceCorrect_g ', 2 * diceCorrect_g[0].item()])
            log.debug(['Prediction + Label      ', dicePrediction_g[0].item()])
            log.debug(['diceLoss_g      ', diceLoss_g[0].item()])
            assert False

        return diceLoss_g


    def logImages(self, epoch_ndx, mode_str, dl):
        images_iter = sorted(dl.dataset.series_list)[:12]
        for series_ndx, series_uid in enumerate(images_iter):
            ct = getCt(series_uid)

            for slice_ndx in range(6):
                ct_ndx = slice_ndx * ct.hu_a.shape[0] // 5
                ct_ndx = min(ct_ndx, ct.hu_a.shape[0] - 1)
                sample_tup = dl.dataset[(series_uid, ct_ndx, False)]

                ct_t, nodule_t, _, ben_t, mal_t, _, _ = sample_tup

                ct_t[:-1,:,:] += 1
                ct_t[:-1,:,:] /= 2

                input_g = ct_t.to(self.device)
                label_g = nodule_t.to(self.device)

                prediction_g = self.model(input_g.unsqueeze(0))[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()
                label_a = nodule_t.numpy()
                ben_a = ben_t.numpy()
                mal_a = mal_t.numpy()

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                image_a[:,:,0] += prediction_a[0] * (1 - label_a[0])
                image_a[:,:,1] += prediction_a[0] * mal_a[0]
                image_a[:,:,2] += prediction_a[0] * ben_a[0]
                image_a *= 0.5
                image_a[image_a < 0] = 0
                image_a[image_a > 1] = 1

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    '{}/{}_prediction_{}'.format(
                        mode_str,
                        series_ndx,
                        slice_ndx,
                    ),
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )

                # self.diceLoss(label_g, prediction_g, p=True)

                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                    image_a[:,:,0] += (1 - label_a[0]) * ct_t[-1].numpy() # Red
                    image_a[:,:,1] += mal_a[0]  # Green
                    image_a[:,:,2] += ben_a[0]  # Blue

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_ndx,
                            slice_ndx,
                        ),
                        image_a,
                        self.totalTrainingSamples_count,
                        dataformats='HWC',
                    )


    def logMetrics(self,
        epoch_ndx,
        mode_str,
        metrics_t,
    ):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.cpu().detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        malLabel_mask = (metrics_a[METRICS_LABEL_NDX] == 1) | (metrics_a[METRICS_LABEL_NDX] == 3)

        # allLabel_mask = (metrics_a[METRICS_LABEL_NDX] == 2) | (metrics_a[METRICS_LABEL_NDX] == 3)

        allLabel_count = sum_a[METRICS_ATP_NDX] + sum_a[METRICS_AFN_NDX]
        malLabel_count = sum_a[METRICS_MTP_NDX] + sum_a[METRICS_MFN_NDX]

        # allCorrect_count = sum_a[METRICS_ATP_NDX]
        # malCorrect_count = sum_a[METRICS_MTP_NDX]
#
#             falsePos_count = allLabel_count - allCorrect_count
#             falseNeg_count = malLabel_count - malCorrect_count


        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/mal'] = np.nan_to_num(metrics_a[METRICS_MAL_LOSS_NDX, malLabel_mask].mean())
        # metrics_dict['loss/all'] = metrics_a[METRICS_ALL_LOSS_NDX, allLabel_mask].mean()

        # metrics_dict['correct/mal'] = sum_a[METRICS_MTP_NDX] / (sum_a[METRICS_MTP_NDX] + sum_a[METRICS_MFN_NDX]) * 100
        # metrics_dict['correct/all'] = sum_a[METRICS_ATP_NDX] / (sum_a[METRICS_ATP_NDX] + sum_a[METRICS_AFN_NDX]) * 100

        metrics_dict['percent_all/tp'] = sum_a[METRICS_ATP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_a[METRICS_AFN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = sum_a[METRICS_AFP_NDX] / (allLabel_count or 1) * 100

        metrics_dict['percent_mal/tp'] = sum_a[METRICS_MTP_NDX] / (malLabel_count or 1) * 100
        metrics_dict['percent_mal/fn'] = sum_a[METRICS_MFN_NDX] / (malLabel_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_ATP_NDX] \
            / ((sum_a[METRICS_ATP_NDX] + sum_a[METRICS_AFP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_ATP_NDX] \
            / ((sum_a[METRICS_ATP_NDX] + sum_a[METRICS_AFN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
                 # + "{correct/all:-5.1f}% correct ({allCorrect_count:} of {allLabel_count:})"
        ).format(
            epoch_ndx,
            mode_str + '_all',
            # allCorrect_count=allCorrect_count,
            # allLabel_count=allLabel_count,
            **metrics_dict,
        ))

        log.info(("E{} {:8} "
                  + "{loss/mal:.4f} loss, "
                  + "{percent_mal/tp:-5.1f}% tp, {percent_mal/fn:-5.1f}% fn"
                 # + "{correct/mal:-5.1f}% correct ({malCorrect_count:} of {malLabel_count:})"
        ).format(
            epoch_ndx,
            mode_str + '_mal',
            # malCorrect_count=malCorrect_count,
            # malLabel_count=malLabel_count,
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        score = 1 \
            - metrics_dict['loss/mal'] \
            + metrics_dict['pr/f1_score'] \
            - metrics_dict['pr/recall'] * 0.01 \
            - metrics_dict['loss/all']  * 0.0001

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
    #                 # metrics_a[METRICS_PRED_NDX, benHist_mask],
    #                 self.totalTrainingSamples_count,
    #                 # bins=bins,
    #             )
    #
    #             # print name, param.data

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )

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
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            file_path = os.path.join(
                'data-unversioned',
                'part2',
                'models',
                self.cli_args.tb_prefix,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.cli_args.comment,
                    'best',
                )
            )
            torch.save(state, file_path)

            log.debug("Saved model params to {}".format(file_path))


if __name__ == '__main__':
    sys.exit(LunaTrainingApp().main() or 0)
