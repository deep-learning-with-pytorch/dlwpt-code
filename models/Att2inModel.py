# This file contains att2in model
# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel

class Att2inCore(nn.Module):
    def __init__(self, opt):
        super(Att2inCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(state[0][-1])                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class Att2inModel(CaptionModel):
    def __init__(self, opt):
        super(Att2inModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = 1
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.core = Att2inCore(opt)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = []

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.att_feat_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state)
            output = F.log_softmax(self.logit(output))
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
        logprobs = F.log_softmax(self.logit(output))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.att_feat_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, self.fc_feat_size)
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.att_feat_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step

                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state)
            logprobs = F.log_softmax(self.logit(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)