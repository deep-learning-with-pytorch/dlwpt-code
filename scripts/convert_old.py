import argparse
import h5py
import os
import numpy as np
import json


def main(params):
    if not os.path.isdir(params['fc_output_dir']):
        os.mkdir(params['fc_output_dir'])
    if not os.path.isdir(params['att_output_dir']):
        os.mkdir(params['att_output_dir'])

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    with h5py.File(os.path.join(params['fc_output_dir'], 'feats_fc.h5')) as file_fc,\
            h5py.File(os.path.join(params['att_output_dir'], 'feats_att.h5')) as file_att:
        for i, img in enumerate(imgs):
            npy_fc_path = os.path.join(
                params['fc_input_dir'],
                str(img['cocoid']) + '.npy')
            npy_att_path = os.path.join(
                params['att_input_dir'],
                str(img['cocoid']) + '.npz')

            d_set_fc = file_fc.create_dataset(
                str(img['cocoid']), data=np.load(npy_fc_path))
            d_set_att = file_att.create_dataset(
                str(img['cocoid']),
                data=np.load(npy_att_path)['feat'])

            if i % 1000 == 0:
                print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
        file_fc.close()
        file_att.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--fc_output_dir', default='data', help='output directory for fc')
    parser.add_argument('--att_output_dir', default='data', help='output directory for att')
    parser.add_argument('--fc_input_dir', default='data', help='input directory for numpy fc files')
    parser.add_argument('--att_input_dir', default='data', help='input directory for numpy att files')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    main(params)
