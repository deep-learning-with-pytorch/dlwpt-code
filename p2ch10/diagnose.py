import argparse
import datetime
import os
import sys

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import Luna2dSegmentationDataset, TestingLuna2dSegmentationDataset, getCt
from util.logconf import logging
from util.util import xyz2irc
from .model import UNetWrapper

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_tensor/metrics_ary
# METRICS_LABEL_NDX=0
# METRICS_PRED_NDX=1
# METRICS_LOSS_NDX=2
# METRICS_MAL_LOSS_NDX=3
# METRICS_BEN_LOSS_NDX=4
# METRICS_LUNG_LOSS_NDX=5
# METRICS_MASKLOSS_NDX=2
# METRICS_MALLOSS_NDX=3


METRICS_LOSS_NDX = 0
METRICS_LABEL_NDX = 1
METRICS_MFOUND_NDX = 2

METRICS_MOK_NDX = 3
METRICS_MTP_NDX = 4
METRICS_MFN_NDX = 5
METRICS_MFP_NDX = 6
METRICS_BTP_NDX = 7
METRICS_BFN_NDX = 8
METRICS_BFP_NDX = 9

METRICS_MAL_LOSS_NDX = 10
METRICS_BEN_LOSS_NDX = 11
METRICS_SIZE = 12




class LunaDiagnoseApp(object):
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

        parser.add_argument('--series-uid',
            help='Limit inference to this Series UID only.',
            default=None,
            type=str,
        )


        parser.add_argument('segmentation_path',
            help="Path to the saved segmentation model",
            nargs=1,
            default='none',
        )

        parser.add_argument('classification_path',
            help="Path to the saved classification model",
            nargs=1,
        )

        self.cli_args = parser.parse_args(sys_argv)
        # self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.seg_model, self.cls_model = self.initModels()
        # self.optimizer = self.initOptimizer()



    def initModels(self):
        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = UNetWrapper(in_channels=8, n_classes=2, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upconv')
        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        cls_model = UNetWrapper(in_channels=8, n_classes=2, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upconv')
        cls_model.load_state_dict(seg_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model = seg_model.to(self.device)
            cls_model = cls_model.to(self.device)

        return seg_model, cls_model


    def initSegmentationDl(self, epoch_ndx):
        seg_ds = Luna2dSegmentationDataset(
                test_stride=10,
                contextSlices_count=3,
                series_uid=self.cli_args.series_uid,
            )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl

    def initTestDl(self):
        if self.cli_args.segmentation:
            test_ds = TestingLuna2dSegmentationDataset(
                    test_stride=10,
                    isTestSet_bool=True,
                    contextSlices_count=3,

                    # scaled_bool=self.cli_args.scaled or self.cli_args.multiscaled or self.cli_args.augmented,
                    # multiscaled_bool=self.cli_args.multiscaled,
                    # augmented_bool=self.cli_args.augmented,
                )
        else:
            assert False

        test_dl = DataLoader(
            test_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return test_dl


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        seg_dl = self.initSegmentationDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            train_dl = self.initTrainDl(epoch_ndx)

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(test_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trainingMetrics_tensor = self.doTraining(epoch_ndx, train_dl)
            if self.cli_args.segmentation:
                self.logImages(epoch_ndx, train_dl, test_dl)

            testingMetrics_tensor = self.doTesting(epoch_ndx, test_dl)
            self.logMetrics(epoch_ndx, trainingMetrics_tensor, testingMetrics_tensor)

            self.saveModel(epoch_ndx)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.tst_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trainingMetrics_tensor = torch.zeros(METRICS_SIZE, len(train_dl.dataset))
        train_dl.dataset.shuffleSamples()
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            if self.cli_args.segmentation:
                loss_var = self.computeSegmentationLoss(batch_ndx, batch_tup, train_dl.batch_size, trainingMetrics_tensor)
            else:
                loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trainingMetrics_tensor)

            if loss_var is not None:
                loss_var.backward()
                self.optimizer.step()
            del loss_var

        self.totalTrainingSamples_count += trainingMetrics_tensor.size(1)

        return trainingMetrics_tensor

    def doTesting(self, epoch_ndx, test_dl):
        with torch.no_grad():
            self.model.eval()
            testingMetrics_tensor = torch.zeros(METRICS_SIZE, len(test_dl.dataset))
            batch_iter = enumerateWithEstimate(
                test_dl,
                "E{} Testing ".format(epoch_ndx),
                start_ndx=test_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                if self.cli_args.segmentation:
                    self.computeSegmentationLoss(batch_ndx, batch_tup, test_dl.batch_size, testingMetrics_tensor)
                else:
                    self.computeBatchLoss(batch_ndx, batch_tup, test_dl.batch_size, testingMetrics_tensor)

        return testingMetrics_tensor

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

    def computeSegmentationLoss(self, batch_ndx, batch_tup, batch_size, metrics_tensor):
        input_tensor, label_tensor, _series_list, _start_list = batch_tup

        # if label_tensor.max() < 0.5:
        #     return None

        input_devtensor = input_tensor.to(self.device)
        label_devtensor = label_tensor.to(self.device)

        prediction_devtensor = self.model(input_devtensor)

        # assert prediction_devtensor.is_contiguous()

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_tensor.size(0)
        max2 = lambda t: t.view(t.size(0), -1).max(dim=1)[0]
        intersectionSum = lambda a, b: (a * b.to(torch.float32)).view(a.size(0), -1).sum(dim=1)

        diceLoss_devtensor = self.diceLoss(label_devtensor, prediction_devtensor)

        with torch.no_grad():

            boolPrediction_tensor = prediction_devtensor.to('cpu') > 0.5

            metrics_tensor[METRICS_LABEL_NDX, start_ndx:end_ndx] = max2(label_tensor[:,0])
            metrics_tensor[METRICS_MFOUND_NDX, start_ndx:end_ndx] = (max2(label_tensor[:, 0] * boolPrediction_tensor[:, 1].to(torch.float32)) > 0.5)

            metrics_tensor[METRICS_MOK_NDX, start_ndx:end_ndx] = intersectionSum( label_tensor[:,0],  torch.max(boolPrediction_tensor, dim=1)[0])

            metrics_tensor[METRICS_MTP_NDX, start_ndx:end_ndx] = intersectionSum( label_tensor[:,0],  boolPrediction_tensor[:,0])
            metrics_tensor[METRICS_MFN_NDX, start_ndx:end_ndx] = intersectionSum( label_tensor[:,0], ~boolPrediction_tensor[:,0])
            metrics_tensor[METRICS_MFP_NDX, start_ndx:end_ndx] = intersectionSum(1 - label_tensor[:,0],  boolPrediction_tensor[:,0])

            metrics_tensor[METRICS_BTP_NDX, start_ndx:end_ndx] = intersectionSum( label_tensor[:,1],  boolPrediction_tensor[:,1])
            metrics_tensor[METRICS_BFN_NDX, start_ndx:end_ndx] = intersectionSum( label_tensor[:,1], ~boolPrediction_tensor[:,1])
            metrics_tensor[METRICS_BFP_NDX, start_ndx:end_ndx] = intersectionSum(1 - label_tensor[:,1],  boolPrediction_tensor[:,1])

            diceLoss_tensor = diceLoss_devtensor.to('cpu')
            metrics_tensor[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_tensor

            malLoss_devtensor = self.diceLoss(label_devtensor[:,0], prediction_devtensor[:,0])
            malLoss_tensor = malLoss_devtensor.to('cpu')#.unsqueeze(1)
            metrics_tensor[METRICS_MAL_LOSS_NDX, start_ndx:end_ndx] = malLoss_tensor

            benLoss_devtensor = self.diceLoss(label_devtensor[:,1], prediction_devtensor[:,1])
            benLoss_tensor = benLoss_devtensor.to('cpu')#.unsqueeze(1)
            metrics_tensor[METRICS_BEN_LOSS_NDX, start_ndx:end_ndx] = benLoss_tensor

            # lungLoss_devtensor = self.diceLoss(label_devtensor[:,2], prediction_devtensor[:,2])
            # lungLoss_tensor = lungLoss_devtensor.to('cpu').unsqueeze(1)
            # metrics_tensor[METRICS_LUNG_LOSS_NDX, start_ndx:end_ndx] = lungLoss_tensor

        # TODO: replace with torch.autograd.detect_anomaly
        # assert np.isfinite(metrics_tensor).all()

        # return nn.MSELoss()(prediction_devtensor, label_devtensor)

        return diceLoss_devtensor.mean()
        # return self.diceLoss(label_devtensor[:,0], prediction_devtensor[:,0]).mean()

    def diceLoss(self, label_devtensor, prediction_devtensor, epsilon=0.01):
        # sum2 = lambda t: t.sum([1,2,3,4])
        sum2 = lambda t: t.view(t.size(0), -1).sum(dim=1)
        # max2 = lambda t: t.view(t.size(0), -1).max(dim=1)[0]

        diceCorrect_devtensor = sum2(prediction_devtensor * label_devtensor)
        dicePrediction_devtensor = sum2(prediction_devtensor)
        diceLabel_devtensor = sum2(label_devtensor)
        epsilon_devtensor = torch.ones_like(diceCorrect_devtensor) * epsilon
        diceLoss_devtensor = 1 - (2 * diceCorrect_devtensor + epsilon_devtensor) / (dicePrediction_devtensor + diceLabel_devtensor + epsilon_devtensor)

        return diceLoss_devtensor



    def logImages(self, epoch_ndx, train_dl, test_dl):
        if epoch_ndx > 0: # TODO revert
            self.initTensorboardWriters()

            for mode_str, dl in [('trn', train_dl), ('tst', test_dl)]:
                for i, series_uid in enumerate(sorted(dl.dataset.series_list)[:12]):
                    ct = getCt(series_uid)
                    noduleInfo_tup = (ct.malignantInfo_list or ct.benignInfo_list)[0]
                    center_irc = xyz2irc(noduleInfo_tup.center_xyz, ct.origin_xyz, ct.vxSize_xyz, ct.direction_tup)

                    sample_tup = dl.dataset[(series_uid, int(center_irc.index))]
                    input_tensor = sample_tup[0].unsqueeze(0)
                    label_tensor = sample_tup[1].unsqueeze(0)

                    input_devtensor = input_tensor.to(self.device)
                    label_devtensor = label_tensor.to(self.device)

                    prediction_devtensor = self.model(input_devtensor)
                    prediction_ary = prediction_devtensor.to('cpu').detach().numpy()

                    image_ary = np.zeros((512, 512, 3), dtype=np.float32)
                    image_ary[:,:,:] = (input_tensor[0,2].numpy().reshape((512,512,1))) * 0.25
                    image_ary[:,:,0] += prediction_ary[0,0] * 0.5
                    image_ary[:,:,1] += prediction_ary[0,1] * 0.25
                    # image_ary[:,:,2] += prediction_ary[0,2] * 0.5

                    # log.debug([image_ary.__array_interface__['typestr']])

                    # image_ary = (image_ary * 255).astype(np.uint8)

                    # log.debug([image_ary.__array_interface__['typestr']])

                    writer = getattr(self, mode_str + '_writer')
                    writer.add_image('{}/{}_pred'.format(mode_str, i), image_ary, self.totalTrainingSamples_count)

                    if epoch_ndx == 1:
                        label_ary = label_tensor.numpy()

                        image_ary = np.zeros((512, 512, 3), dtype=np.float32)
                        image_ary[:,:,:] = (input_tensor[0,2].numpy().reshape((512,512,1))) * 0.25
                        image_ary[:,:,0] += label_ary[0,0] * 0.5
                        image_ary[:,:,1] += label_ary[0,1] * 0.25
                        image_ary[:,:,2] += (input_tensor[0,-1].numpy() - (label_ary[0,0].astype(np.bool) | label_ary[0,1].astype(np.bool))) * 0.25

                        # log.debug([image_ary.__array_interface__['typestr']])

                        image_ary = (image_ary * 255).astype(np.uint8)

                        # log.debug([image_ary.__array_interface__['typestr']])

                        writer = getattr(self, mode_str + '_writer')
                        writer.add_image('{}/{}_label'.format(mode_str, i), image_ary, self.totalTrainingSamples_count)


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
            metrics_ary = metrics_tensor.cpu().detach().numpy()
            sum_ary = metrics_ary.sum(axis=1)
            assert np.isfinite(metrics_ary).all()

            malLabel_mask = metrics_ary[METRICS_LABEL_NDX] > classificationThreshold_float
            malFound_mask = metrics_ary[METRICS_MFOUND_NDX] > classificationThreshold_float

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
            metrics_dict['loss/mal'] = metrics_ary[METRICS_MAL_LOSS_NDX].mean()
            metrics_dict['loss/ben'] = metrics_ary[METRICS_BEN_LOSS_NDX].mean()

            metrics_dict['flagged/all'] = sum_ary[METRICS_MOK_NDX] / (sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFN_NDX]) * 100
            metrics_dict['flagged/slices'] = (malLabel_mask & malFound_mask).sum() / malLabel_mask.sum() * 100

            metrics_dict['correct/mal'] = sum_ary[METRICS_MTP_NDX] / (sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFN_NDX]) * 100
            metrics_dict['correct/ben'] = sum_ary[METRICS_BTP_NDX] / (sum_ary[METRICS_BTP_NDX] + sum_ary[METRICS_BFN_NDX]) * 100

            precision = metrics_dict['pr/precision'] = sum_ary[METRICS_MTP_NDX] / ((sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFP_NDX]) or 1)
            recall    = metrics_dict['pr/recall']    = sum_ary[METRICS_MTP_NDX] / ((sum_ary[METRICS_MTP_NDX] + sum_ary[METRICS_MFN_NDX]) or 1)

            metrics_dict['pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

            log.info(("E{} {:8} "
                     + "{loss/all:.4f} loss, "
                     + "{flagged/all:-5.1f}% pixels flagged, "
                     + "{flagged/slices:-5.1f}% slices flagged, "
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
                mode_str + '_msk',
                benCorrect_count=benCorrect_count,
                benLabel_count=benLabel_count,
                **metrics_dict,
            ))

            if epoch_ndx > 0: # TODO revert
                self.initTensorboardWriters()
                writer = getattr(self, mode_str + '_writer')

                for key, value in metrics_dict.items():
                    writer.add_scalar('seg_' + key, value, self.totalTrainingSamples_count)

#                 writer.add_pr_curve(
#                     'pr',
#                     metrics_ary[METRICS_LABEL_NDX],
#                     metrics_ary[METRICS_PRED_NDX],
#                     self.totalTrainingSamples_count,
#                 )

#                 benHist_mask = benLabel_mask & (metrics_ary[METRICS_PRED_NDX] > 0.01)
#                 malHist_mask = malLabel_mask & (metrics_ary[METRICS_PRED_NDX] < 0.99)
#
#                 bins = [x/50.0 for x in range(51)]
#                 writer.add_histogram(
#                     'is_ben',
#                     metrics_ary[METRICS_PRED_NDX, benHist_mask],
#                     self.totalTrainingSamples_count,
#                     bins=bins,
#                 )
#                 writer.add_histogram(
#                     'is_mal',
#                     metrics_ary[METRICS_PRED_NDX, malHist_mask],
#                     self.totalTrainingSamples_count,
#                     bins=bins,
#                 )

    def saveModel(self, epoch_ndx):
        file_path = os.path.join('data', 'models', self.cli_args.tb_prefix, '{}_{}.{}.state'.format(self.time_str, self.cli_args.comment, self.totalTrainingSamples_count))

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        state = {
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
            # 'resumed_from': self.cli_args.resume,
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))


if __name__ == '__main__':
    sys.exit(LunaDiagnoseApp().main() or 0)
