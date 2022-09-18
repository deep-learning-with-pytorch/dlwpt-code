import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
from .Att2inModel import Att2inModel
from .AttModel import *

def setup(opt):
    
    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'show_attend_tell':
        model = ShowAttendTellModel(opt)
    # img is concatenated with word embedding at every time step as the input of lstm
    elif opt.caption_model == 'all_img':
        model = AllImgModel(opt)
    # FC model in self-critical
    elif opt.caption_model == 'fc':
        model = FCModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model