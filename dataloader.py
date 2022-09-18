from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch.utils.data as data

import multiprocessing


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split,
                                                    self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def read_files(self):
        self.feats_fc = h5py.File(os.path.join(
            self.opt.input_fc_dir, 'feats_fc.h5'), 'r')
        self.feats_att = h5py.File(os.path.join(
            self.opt.input_att_dir, 'feats_att.h5'), 'r')

    def get_data(self, ix):
        self.read_files()
        index = str(self.info['images'][ix]['id'])
        if self.use_att:
            return (np.array(self.feats_fc[index]).astype('float32'),
                    np.array(self.feats_att[index]).astype('float32'), ix)
        else:
            return (np.array(self.feats_fc[index]).astype('float32'),
                    np.zeros((1, 1, 1)).astype('float32'), ix)

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.use_att = getattr(opt, 'use_att', True)

        # load json file which contains additional information about dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir,
              opt.input_att_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r',
                                       driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split,
                                                        self,
                                                        split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = []
        att_batch = []
        label_batch = np.zeros(
            [batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros(
            [batch_size * seq_per_img, self.seq_length + 2], dtype='float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att,\
                ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch += [tmp_fc] * seq_per_img
            att_batch += [tmp_att] * seq_per_img

            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1  # number of captions available for this image
            assert ncap > 0, 'an image does not have any label.'

            if ncap < seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
                for q in range(seq_per_img):
                    ixl = random.randint(ix1, ix2)
                    seq[q, :] = self.h5_label_file['labels'][ixl,
                                                             :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - seq_per_img + 1)
                seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img,
                                                   :self.seq_length]

            label_batch[i * seq_per_img: (i + 1) * seq_per_img,
                        1: self.seq_length + 1] = seq

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(
                self.h5_label_file['labels'][self.label_start_ix[ix] - 1:
                                             self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        data['fc_feats'] = np.stack(fc_batch)
        data['att_feats'] = np.stack(att_batch)
        data['labels'] = label_batch
        data['gts'] = gts
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': len(self.split_ix[split]),
                          'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset,
    # but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according
    # the index. However, it's minimum change to switch to pytorch data loading
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        return self.get_data(ix)

    def __len__(self):
        return len(self.info['images'])


class ArraySampler(data.sampler.SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name,
        caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases:
        1. not hasattr(self, 'split_loader'): Resume from previous training.
        Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in
         the get_minibatch_inds already.
        """
        # batch_size is 0, the merge is done in DataLoader class
        sampler = ArraySampler(
            self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:])
        self.split_loader = iter(
            data.DataLoader(dataset=self.dataloader,
                            batch_size=1,
                            sampler=sampler,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=multiprocessing.cpu_count(),
                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()
        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]
