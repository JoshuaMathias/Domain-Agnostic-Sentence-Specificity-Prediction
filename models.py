
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf
"""

import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
from dotenv import load_dotenv


# Get .env variables:
load_dotenv()
USE_CUDA = os.getenv('USE_CUDA')
# Convert to bool, but only if .env file exists:
if USE_CUDA is not None:
    USE_CUDA = USE_CUDA == 'True'

"""
BLSTM (max/mean) encoder
"""

class BLSTMEncoder(nn.Module):

    def __init__(self, config):
        super(BLSTMEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.n_enc_layers,
                                bidirectional=True, dropout=self.dpout_model)

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        # If .env file exists, use env variable for `use_cuda`:
        if USE_CUDA is not None:
            use_cuda = USE_CUDA
        else:
            use_cuda = self.use_cuda

        if use_cuda:
            idx_sort = torch.from_numpy(idx_sort).cuda()
            sent = sent.index_select(1, Variable(idx_sort).cuda())
        else:
            idx_sort = torch.from_numpy(idx_sort)
            sent = sent.index_select(1, Variable(idx_sort))


        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len.copy())
        self.enc_lstm.flatten_parameters()
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        if use_cuda:
            idx_unsort = torch.from_numpy(idx_unsort).cuda()
            sent_output = sent_output.index_select(1, Variable(idx_unsort)).cuda()
        else:
            idx_unsort = torch.from_numpy(idx_unsort)
            sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1)
            if use_cuda:
                sent_len = sent_len.cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def set_glove_path(self, glove_path):
        self.glove_path = glove_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [s.split() if not tokenize else word_tokenize(s)
                     for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        return word_dict

    def get_glove(self, word_dict):
        assert hasattr(self, 'glove_path'), \
               'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))
        return word_vec

    def get_glove_k(self, K):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        # create word_vec with k first glove vectors
        k = 0
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in ['<s>', '</s>']:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in ['<s>', '</s>']]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

    # build GloVe vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        self.word_vec = self.get_glove_k(K)
        print('Vocab size : {0}'.format(K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_glove(word_dict)
            self.word_vec.update(new_word_vec)
        print('New vocab size : {0} (added {1} words)'.format(
                        len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
                     ['<s>']+word_tokenize(s)+['</s>'] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
                               Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : {0}/{1} ({2} %)'.format(
                        n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = Variable(self.get_batch(
                        sentences[stidx:stidx + bsize]), volatile=True)
            if self.use_cuda:
               batch = batch.cuda()
            batch = self.forward(
                (batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : {0} sentences/s ({1} mode, bsize={2})'.format(
                    round(len(embeddings)/(time.time()-tic), 2),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    def visualize(self, sent, tokenize=True):
        if tokenize:
            from nltk.tokenize import word_tokenize

        sent = sent.split() if not tokenize else word_tokenize(sent)
        sent = [['<s>'] + [word for word in sent if word in self.word_vec] +
                ['</s>']]

        if ' '.join(sent[0]) == '<s> </s>':
            import warnings
            warnings.warn('No words in "{0}" have glove vectors. Replacing \
                           by "<s> </s>"..'.format(sent))
        batch = Variable(self.get_batch(sent), volatile=True)

        if self.is_cuda():
            batch = batch.cuda()
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        # output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0*n/np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        return output, idxs

"""
BiGRU encoder (first/last hidden states)
"""


class BGRUlastEncoder(nn.Module):
    def __init__(self, config):
        super(BGRUlastEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.GRU(self.word_emb_dim, self.enc_lstm_dim, self.n_enc_layers,
                               bidirectional=True, dropout=self.dpout_model)
        self.init_lstm = Variable(torch.FloatTensor(2, self.bsize,
                                  self.enc_lstm_dim).zero_())
        if self.use_cuda:
            self.init_lstm = self.init_lstm.cuda()

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)

        self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else \
                Variable(torch.FloatTensor(2, bsize, self.enc_lstm_dim).zero_())
        if self.usa_cuda:
            self.init_lstm = self.init_lstm.cuda()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        if self.use_cuda:
            sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)).cuda())
        else:
            sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        _, hn = self.enc_lstm(sent_packed, self.init_lstm)
        emb = torch.cat((hn[0], hn[1]), 1)  # batch x 2*nhid

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        if self.use_cuda:
            emb = emb.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)).cuda())
        else:
            emb = emb.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)))

        return emb


"""
BLSTM encoder with projection after BiLSTM
"""


class BLSTMprojEncoder(nn.Module):
    def __init__(self, config):
        super(BLSTMprojEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.n_enc_layers,
                                bidirectional=True, dropout=self.dpout_model)
        self.init_lstm = Variable(torch.FloatTensor(2, self.bsize,
                                  self.enc_lstm_dim).zero_()).cuda()
        self.proj_enc = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                  bias=False)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)

        self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else \
                Variable(torch.FloatTensor(2, bsize, self.enc_lstm_dim).zero_()).cuda()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)).cuda())

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed,
                                    (self.init_lstm, self.init_lstm))[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1,
                Variable(torch.cuda.LongTensor(idx_unsort)).cuda())

        sent_output = self.proj_enc(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(-1, bsize, 2*self.enc_lstm_dim)
        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0].squeeze(0)

        return emb


"""
LSTM encoder
"""


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.n_enc_layers,
                                bidirectional=False, dropout=self.dpout_model)
        self.init_lstm = Variable(torch.FloatTensor(1, self.bsize,
            self.enc_lstm_dim).zero_()).cuda()

    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch) | sent Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)

        self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else \
                Variable(torch.FloatTensor(1, bsize, self.enc_lstm_dim).zero_()).cuda()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort))).cuda()

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed, (self.init_lstm,
                      self.init_lstm))[1][0].squeeze(0)  # batch x 2*nhid

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        emb = sent_output.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)).cuda())

        return emb


"""
GRU encoder
"""


class GRUEncoder(nn.Module):
    def __init__(self, config):
        super(GRUEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim =  config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.GRU(self.word_emb_dim, self.enc_lstm_dim, self.n_enc_layers,
                               bidirectional=False, dropout=self.dpout_model)
        self.init_lstm = Variable(torch.FloatTensor(1, self.bsize,
            self.enc_lstm_dim).zero_()).cuda()

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)

        self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else \
                Variable(torch.FloatTensor(1, bsize, self.enc_lstm_dim).zero_()).cuda()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)).cuda())

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)

        sent_output = self.enc_lstm(sent_packed, self.init_lstm)[1].squeeze(0)
        # batch x 2*nhid

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        emb = sent_output.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)).cuda())

        return emb


"""
Inner attention from "hierarchical attention for document classification"
"""


class InnerAttentionNAACLEncoder(nn.Module):
    def __init__(self, config):
        super(InnerAttentionNAACLEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.n_enc_layers,
                                bidirectional=True)
        self.init_lstm = Variable(torch.FloatTensor(2, self.bsize,
                                  self.enc_lstm_dim).zero_()).cuda()

        self.proj_key = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                  bias=False)
        self.proj_lstm = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                   bias=False)
        self.query_embedding = nn.Embedding(1, 2*self.enc_lstm_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)

        self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else \
                Variable(torch.FloatTensor(2, bsize, self.enc_lstm_dim).zero_()).cuda()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)).cuda())
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        self.enc_lstm.flatten_parameters()
        sent_output = self.enc_lstm(sent_packed,
                                    (self.init_lstm, self.init_lstm))[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1, Variable(torch.cuda.LongTensor(idx_unsort)).cuda())

        sent_output = sent_output.transpose(0,1).contiguous()

        sent_output_proj = self.proj_lstm(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)

        sent_key_proj = self.proj_key(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)

        sent_key_proj = torch.tanh(sent_key_proj)
        # NAACL paper: u_it=tanh(W_w.h_it + b_w)  (bsize, seqlen, 2nhid)

        sent_w = self.query_embedding(Variable(torch.LongTensor(bsize*[0]).cuda())).unsqueeze(2) #(bsize, 2*nhid, 1)

        Temp = 2
        keys = sent_key_proj.bmm(sent_w).squeeze(2) / Temp

        # Set probas of padding to zero in softmax
        keys = keys + ((keys == 0).float()*-10000)
       # print(keys/Temp)
        alphas = self.softmax(keys/Temp).unsqueeze(2).expand_as(sent_output)
        #if int(time.time()) % 100 == 0:
        #    print('w', torch.max(sent_w), torch.min(sent_w))
        #    print('alphas', alphas[0, :, 0])
        atto=self.softmax(keys/Temp)
        emb = torch.sum(alphas * sent_output_proj, 1).squeeze(1)

        return emb


"""
Inner attention inspired from "Self-attentive ..."
"""


class InnerAttentionMILAEncoder(nn.Module):
    def __init__(self, config):
        super(InnerAttentionMILAEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim =  config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.n_enc_layers,
                                bidirectional=True)
        self.init_lstm = Variable(torch.FloatTensor(2, self.bsize,
                                  self.enc_lstm_dim).zero_()).cuda()

        self.proj_key = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                  bias=False)
        self.proj_lstm = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                   bias=False)
        self.query_embedding = nn.Embedding(2, 2*self.enc_lstm_dim)
        self.softmax = nn.Softmax()

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)

        self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else \
                Variable(torch.FloatTensor(2, bsize, self.enc_lstm_dim).zero_()).cuda()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)).cuda())
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed,
                                    (self.init_lstm, self.init_lstm))[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1,
            Variable(torch.cuda.LongTensor(idx_unsort)).cuda())

        sent_output = sent_output.transpose(0,1).contiguous()
        sent_output_proj = self.proj_lstm(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)
        sent_key_proj = self.proj_key(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)
        sent_key_proj = torch.tanh(sent_key_proj)
        # NAACL : u_it=tanh(W_w.h_it + b_w) like in NAACL paper

        # Temperature
        Temp = 3

        sent_w1 = self.query_embedding(Variable(torch.LongTensor(bsize*[0]).cuda())).unsqueeze(2) #(bsize, nhid, 1)
        keys1 = sent_key_proj.bmm(sent_w1).squeeze(2) / Temp
        keys1 = keys1 + ((keys1 == 0).float()*-1000)
        alphas1 = self.softmax(keys1).unsqueeze(2).expand_as(sent_key_proj)
        emb1 = torch.sum(alphas1 * sent_output_proj, 1).squeeze(1)


        sent_w2 = self.query_embedding(Variable(torch.LongTensor(bsize*[1]).cuda())).unsqueeze(2) #(bsize, nhid, 1)
        keys2 = sent_key_proj.bmm(sent_w2).squeeze(2) / Temp
        keys2 = keys2 + ((keys2 == 0).float()*-1000)
        alphas2 = self.softmax(keys2).unsqueeze(2).expand_as(sent_key_proj)
        emb2 = torch.sum(alphas2 * sent_output_proj, 1).squeeze(1)

        sent_w3 = self.query_embedding(Variable(torch.LongTensor(bsize*[1]).cuda())).unsqueeze(2) #(bsize, nhid, 1)
        keys3 = sent_key_proj.bmm(sent_w3).squeeze(2) / Temp
        keys3 = keys3 + ((keys3 == 0).float()*-1000)
        alphas3 = self.softmax(keys3).unsqueeze(2).expand_as(sent_key_proj)
        emb3 = torch.sum(alphas3 * sent_output_proj, 1).squeeze(1)

        sent_w4 = self.query_embedding(Variable(torch.LongTensor(bsize*[1]).cuda())).unsqueeze(2) #(bsize, nhid, 1)
        keys4 = sent_key_proj.bmm(sent_w4).squeeze(2) / Temp
        keys4 = keys4 + ((keys4 == 0).float()*-1000)
        alphas4 = self.softmax(keys4).unsqueeze(2).expand_as(sent_key_proj)
        emb4 = torch.sum(alphas4 * sent_output_proj, 1).squeeze(1)


        if int(time.time()) % 100 == 0:
            print('alphas', torch.cat((alphas1.data[0, :, 0],
                                       alphas2.data[0, :, 0],
                                       torch.abs(alphas1.data[0, :, 0] -
                                                 alphas2.data[0, :, 0])), 1))

        emb = torch.cat((emb1, emb2, emb3, emb4), 1)
        return emb


"""
Inner attention from Yang et al.
"""


class InnerAttentionYANGEncoder(nn.Module):
    def __init__(self, config):
        super(InnerAttentionYANGEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.n_enc_layers,
                                bidirectional=True)
        self.init_lstm = Variable(torch.FloatTensor(2, self.bsize,
            self.enc_lstm_dim).zero_()).cuda()

        self.proj_lstm = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                   bias=True)
        self.proj_query = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                    bias=True)
        self.proj_enc = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                  bias=True)

        self.query_embedding = nn.Embedding(1, 2*self.enc_lstm_dim)
        self.softmax = nn.Softmax()

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)

        self.init_lstm = self.init_lstm if bsize == self.init_lstm.size(1) else \
                Variable(torch.FloatTensor(2, bsize, self.enc_lstm_dim).zero_()).cuda()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, Variable(torch.cuda.LongTensor(idx_sort)).cuda())
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed,
                                    (self.init_lstm, self.init_lstm))[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1,
            Variable(torch.cuda.LongTensor(idx_unsort)).cuda())

        sent_output = sent_output.transpose(0,1).contiguous()

        sent_output_proj = self.proj_lstm(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)

        sent_keys = self.proj_enc(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)

        sent_max = torch.max(sent_output, 1)[0].squeeze(1)  # (bsize, 2*nhid)
        sent_summary = self.proj_query(
                       sent_max).unsqueeze(1).expand_as(sent_keys)
        # (bsize, seqlen, 2*nhid)

        sent_M = torch.tanh(sent_keys + sent_summary)
        # (bsize, seqlen, 2*nhid) YANG : M = tanh(Wh_i + Wh_avg
        sent_w = self.query_embedding(Variable(torch.LongTensor(
            bsize*[0]).cuda())).unsqueeze(2)  # (bsize, 2*nhid, 1)

        sent_alphas = self.softmax(sent_M.bmm(sent_w).squeeze(2)).unsqueeze(1)
        # (bsize, 1, seqlen)

        if int(time.time()) % 200 == 0:
            print('w', torch.max(sent_w[0]), torch.min(sent_w[0]))
            print('alphas', sent_alphas[0][0][0:sent_len[0]])
        # Get attention vector
        emb = sent_alphas.bmm(sent_output_proj).squeeze(1)

        return emb



"""
Hierarchical ConvNet
"""
class ConvNetEncoder(nn.Module):
    def __init__(self, config):
        super(ConvNetEncoder, self).__init__()

        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.convnet1 = nn.Sequential(
            nn.Conv1d(self.word_emb_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet2 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet3 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet4 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )



    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0,1).transpose(1,2).contiguous()
        # batch, nhid, seqlen)

        sent = self.convnet1(sent)
        u1 = torch.max(sent, 2)[0]

        sent = self.convnet2(sent)
        u2 = torch.max(sent, 2)[0]

        sent = self.convnet3(sent)
        u3 = torch.max(sent, 2)[0]

        sent = self.convnet4(sent)
        u4 = torch.max(sent, 2)[0]

        emb = torch.cat((u1, u2, u3, u4), 1)

        return emb


"""
Main module for Natural Language Inference
"""


class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 4*2*self.enc_lstm_dim
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb
class PDTBNet(nn.Module):
    def __init__(self, config):
        super(PDTBNet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 2*self.enc_lstm_dim+14
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        self.bn=nn.BatchNorm1d(14)

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

  #  def forward(self, s1, s2):
    def forward(self, s1,ss):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        #v = self.encoder(s2)
        features = torch.cat((u,self.bn(ss)), 1)
        output = self.classifier(features)
        #output = self.classifier(u)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb
class PDTBNee(nn.Module):
    def __init__(self, config,we):
        super(PDTBNee, self).__init__()
        self.embeddings=nn.Embedding(config['n_words'],config['word_emb_dim'])
        self.embeddings.weight=nn.Parameter(we)
        #print(self.embeddings(Variable(torch.LongTensor([0]))))
        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 2*self.enc_lstm_dim+14
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        self.bn=nn.BatchNorm1d(14)

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

  #  def forward(self, s1, s2):
    def forward(self, s1,ss):
        # s1 : (s1, s1_len)

        ss1,ss2=s1
        sss=Variable(torch.FloatTensor(ss1.size(0),ss1.size(1),self.embeddings.weight.size(1))).cuda()
        #sss=torch.index_select(self.embeddings, 2, ss1)
        for i in range(ss1.size(0)):
            for j in range (sss.size(1)):
#                print (type(ss1[i,j,0]))
                #lookup_tensor = torch.LongTensor([0])
                #sss[i,j,:]=self.embeddings(Variable(lookup_tensor))
                sss[i,j,:]=self.embeddings(ss1[i,j,0].long())
        #print(sss)
        #print(ss2)
        #print(ss1)

        u = self.encoder((sss, ss2))
        #v = self.encoder(s2)
        features = torch.cat((u,self.bn(ss)), 1)
        output = self.classifier(features)
        #output = self.classifier(u)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb

class AsNet(nn.Module):
    def __init__(self, config):
        super(AsNet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 2*self.enc_lstm_dim+14
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        self.bn=nn.BatchNorm1d(14)

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

  #  def forward(self, s1, s2):
    def forward(self, s1,ss):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        #v = self.encoder(s2)
        features = torch.cat((u,self.bn(ss)), 1)
        output = self.classifier(features)
        #output = self.classifier(u)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb
class AscNet(nn.Module):
    def __init__(self, config):
        super(AscNet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.dpout_fc = config['dpout_fc']
        self.inputdim = 64

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

  #  def forward(self, s1, s2):
    def forward(self, s1):
        output = self.classifier(s1)
        return output


"""
Main module for Classification
"""


class ClassificationNet(nn.Module):
    def __init__(self, config):
        super(ClassificationNet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 2*self.enc_lstm_dim
        self.inputdim = 4*self.inputdim if self.encoder_type == "ConvNetEncoder" else self.inputdim
        self.inputdim = self.enc_lstm_dim if self.encoder_type =="LSTMEncoder" else self.inputdim
        self.classifier = nn.Sequential(
            nn.Linear(self.inputdim, 512),
            nn.Linear(512, self.n_classes),
        )

    def forward(self, s1):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)

        output = self.classifier(u)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb

class PDTBNetc(nn.Module):
    def __init__(self, config):
        super(PDTBNetc, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 2*self.enc_lstm_dim+14
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        self.bn=nn.BatchNorm1d(14)

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

  #  def forward(self, s1, s2):
    def forward(self, s1,ss,sc,scl):
        # s1 : (s1, s1_len)
        u = self.encoder(s1,sc,scl)
        #v = self.encoder(s2)
        features = torch.cat((u,self.bn(ss)), 1)
        output = self.classifier(features)
        #output = self.classifier(u)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb

class BLSTMEncoderc(nn.Module):

    def __init__(self, config):
        super(BLSTMEncoderc, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.use_cuda = config['use_cuda']

        self.embeddings=nn.Embedding(108,config['char_emb_dim'])
        self.char_lstm = nn.LSTM(config['char_emb_dim'], config['char_rep_dim'], self.n_enc_layers,
                                bidirectional=True, dropout=self.dpout_model)

        self.enc_lstm = nn.LSTM(self.word_emb_dim+2*config['char_rep_dim'], self.enc_lstm_dim, self.n_enc_layers,
                                bidirectional=True, dropout=self.dpout_model)

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

    def forward(self, sent_tuple,sc,sc_len):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        if self.use_cuda:
            idx_sort = torch.from_numpy(idx_sort).cuda()
            sent = sent.index_select(1, Variable(idx_sort))
        else:
            idx_sort = torch.from_numpy(idx_sort).cuda()
            sent = sent.index_select(1, Variable(idx_sort))

#        print('sent')

 #       print(sent)
  #      print('self.embeddings')
   #     print(self.embeddings)
        #sc = torch.index_select(self.embeddings,0, sc)

#        sc=sc.transpose(1,2)
        s00=sc.size(0)
        s01=sc.size(1)
        sc=sc.contiguous().view(s00*s01,-1)
        sc=sc.transpose(0,1)
       # print(sc)
        #print(torch.max(torch.max(sc)))
        #print(torch.min(torch.min(sc)))

        sc=self.embeddings(sc.long())

        uuuua=[]
        for awffwa in range (s00*s01):
            uuuua.append(20)
        sc_len=np.array(uuuua)
        # Handling padding in Recurrent Networks
        sc_packed = nn.utils.rnn.pack_padded_sequence(sc, sc_len)


        self.char_lstm.flatten_parameters()
        char_output = self.char_lstm(sc_packed)[0]  # seqlen x batch x 2*nhid
        char_output = nn.utils.rnn.pad_packed_sequence(char_output)[0]
#30*960*16->30*32*30*16
        char_output=char_output.contiguous().transpose(0,1)
#        char_output=char_output.contiguous().view(s00,s01,char_output.size(1),char_output.size(2))

#        print('char_output@@@@@@@@@@@')
#        print(char_output)

#        sc=sc.contiguous().view(s00,s01,-1)
#        sc=sc.transpose(1,2)
#        print('sent')
#        print(sent)
#        print(Variable(torch.LongTensor(uuuua).cuda()).view(-1,1,1).repeat(1, 1,char_output.size(2))-1)
        if self.use_cuda:
            char_output=torch.gather(char_output,1,Variable(torch.LongTensor(uuuua).cuda()).view(-1,1,1).repeat(1, 1,char_output.size(2))-1)
        else:
            char_output=torch.gather(char_output,1,Variable(torch.LongTensor(uuuua)).view(-1,1,1).repeat(1, 1,char_output.size(2))-1)

        #char_output=char_output.index_select(1,Variable(torch.LongTensor(uuuua).cuda()))
#        print('char_output')
#        print(char_output)

        sent=sent.contiguous().view(s00*s01,sent.size(2))
        sent=torch.cat((sent,char_output.view(char_output.size(0),char_output.size(2))),1)
        sent=sent.contiguous().view(s00,s01,sent.size(1))
#        print('sentt')
#        print(sent)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
#        print('senttt')
#        print(sent_packed)
        self.enc_lstm.flatten_parameters()
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid

        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        if self.use_cuda:
            idx_unsort =  torch.from_numpy(idx_unsort).cuda()
            sent_output = sent_output.index_select(1, Variable(idx_unsort)).cuda()
        else:
            idx_unsort =  torch.from_numpy(idx_unsort)
            sent_output = sent_output.index_select(1, Variable(idx_unsort))


        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def set_glove_path(self, glove_path):
        self.glove_path = glove_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [s.split() if not tokenize else word_tokenize(s)
                     for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        return word_dict

    def get_glove(self, word_dict):
        assert hasattr(self, 'glove_path'), \
               'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))
        return word_vec

    def get_glove_k(self, K):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        # create word_vec with k first glove vectors
        k = 0
        word_vec = {}
        with open(self.glove_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in ['<s>', '</s>']:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in ['<s>', '</s>']]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

    # build GloVe vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        self.word_vec = self.get_glove_k(K)
        print('Vocab size : {0}'.format(K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_glove(word_dict)
            self.word_vec.update(new_word_vec)
        print('New vocab size : {0} (added {1} words)'.format(
                        len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
                     ['<s>']+word_tokenize(s)+['</s>'] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
                               Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : {0}/{1} ({2} %)'.format(
                        n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = Variable(self.get_batch(
                        sentences[stidx:stidx + bsize]), volatile=True)
            if self.is_cuda():
                batch = batch.cuda()
            batch = self.forward(
                (batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : {0} sentences/s ({1} mode, bsize={2})'.format(
                    round(len(embeddings)/(time.time()-tic), 2),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    def visualize(self, sent, tokenize=True):
        if tokenize:
            from nltk.tokenize import word_tokenize

        sent = sent.split() if not tokenize else word_tokenize(sent)
        sent = [['<s>'] + [word for word in sent if word in self.word_vec] +
                ['</s>']]

        if ' '.join(sent[0]) == '<s> </s>':
            import warnings
            warnings.warn('No words in "{0}" have glove vectors. Replacing \
                           by "<s> </s>"..'.format(sent))
        batch = Variable(self.get_batch(sent), volatile=True)

        if self.is_cuda():
            batch = batch.cuda()
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        # output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0*n/np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        return output, idxs
class PDTBNetdd(nn.Module):
    def __init__(self, config):
        super(PDTBNetdd, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.n_enc_layers = config['n_enc_layers']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        # If .env file exists, use env variable:
        if USE_CUDA is not None:
            self.use_cuda = USE_CUDA
        else:
            self.use_cuda = config['use_cuda']

        self.encoder = eval(self.encoder_type)(config)
        dpout_modell=config['dpout_model']
        config['dpout_model']=0

        self.encoder2 = eval(self.encoder_type)(config)
        config['dpout_model']=dpout_modell

        self.inputdim = 2*self.enc_lstm_dim+15
        self.inputdim2 = 2*self.enc_lstm_dim+14
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        self.inputdim2 = 4*self.inputdim2 if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim2
        self.inputdim2 = self.inputdim2/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim2
        self.bn=nn.BatchNorm1d(14)
        self.sig=nn.Sigmoid()

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
            self.classifier2 = nn.Sequential(
                nn.Linear(self.inputdim2, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.BatchNorm1d(self.fc_dim),
                nn.ReLU(),
                nn.Linear(self.fc_dim, 1),
                )

        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

  #  def forward(self, s1, s2):
    def forward(self, s1,ss):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder2(s1)
        vf = torch.cat((v,self.bn(ss)), 1)

        vv = self.classifier2(vf)

        vv=self.sig(vv)
        features = torch.cat((u,self.bn(ss),vv), 1)

        output = self.classifier(features)
        #output = self.classifier(u)
        return output,vv

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb
class disc(nn.Module):
    def __init__(self, config):
        super(disc, self).__init__()
        self.bsize = config['bsize']
        self.f1=nn.Linear(self.bsize,32)
        self.f2=nn.Linear(32,10)
        self.f3=nn.Linear(10,1)
        self.rl=nn.ReLU()

    def forward(self, i):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)
        i=self.f1(i)
        i=self.rl(i)
   #     print(i)
        i=self.f2(i)
        i=self.rl(i)
  #      print(i)
        i=self.f3(i)
        return i
