from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np



device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 10 # maximum sentence length
PAD_token = 0 # used for padding short sentences.
SOS_token = 1 # Start-of-sentence token
EOS_token = 2 # End-of-sentence


# seq2seq model
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        # count PAD, SOS, EOS
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1


    def trim(self, min_count):
        """
        Remove words under a certain threshold.
        """
        if self.trimmed:
            return 
        self.trimmed = True
        keep_words = []
        for k in self.word2count:
            if self.word2count[k] > min_count:
                keep_words.append(k)

        print('Keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) /
            len(self.word2index)
        ))
        
        # reinitialize dicts
        self.word2count = {}
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        for word in keep_words:
            self.addWord(word)
   
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"([^azA-Z.!?]+)", r" ", s)

    return s

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]





class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths)
        
