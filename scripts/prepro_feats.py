"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import h5py
from random import shuffle, seed

import numpy as np
import torch
from torch.autograd import Variable
import skimage.io

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet


def main(params):
  net = getattr(resnet, params['model'])()
  net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
  my_resnet = myResnet(net)
  my_resnet.cuda()
  my_resnet.eval()

  imgs = json.load(open(params['input_json'], 'r'))
  imgs = imgs['images']
  N = len(imgs)

  seed(123) # make reproducible

  dir_fc = params['output_dir']+'_fc'
  dir_att = params['output_dir']+'_att'
  if not os.path.isdir(dir_fc):
    os.mkdir(dir_fc)
  if not os.path.isdir(dir_att):
    os.mkdir(dir_att)

  with h5py.File(os.path.join(dir_fc, 'feats_fc.h5')) as file_fc,\
       h5py.File(os.path.join(dir_att, 'feats_att.h5')) as file_att:
    for i, img in enumerate(imgs):
      # load the image
      I = skimage.io.imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
      # handle grayscale input images
      if len(I.shape) == 2:
        I = I[:,:,np.newaxis]
        I = np.concatenate((I,I,I), axis=2)

      I = I.astype('float32')/255.0
      I = torch.from_numpy(I.transpose([2,0,1])).cuda()
      with torch.no_grad():
          I = Variable(preprocess(I))
          tmp_fc, tmp_att = my_resnet(I, params['att_size'])

      # write to hdf5
      d_set_fc = file_fc.create_dataset(
          str(img['cocoid']), 
          (2048,), dtype="float")
      d_set_att = file_att.create_dataset(
          str(img['cocoid']), 
          (params['att_size'], params['att_size'], 2048), dtype="float")

      d_set_fc[...] = tmp_fc.cpu().float().numpy()
      d_set_att[...] = tmp_att.cpu().float().numpy()
      if i % 1000 == 0:
        print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0 / N))
    file_fc.close()
    file_att.close()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
  parser.add_argument('--output_dir', default='data', help='output directory')

  # options
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
  parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
  parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
