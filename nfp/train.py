#!/usr/bin/env python
"""Neural Fingerprints (NFP)

This is for an experiment of NFP by a conventional approach.

There are two NNs:
- NFP NN: Learns the weights for generating NFP.
- Classifier: Classifier for QSAR.
There are two phase:
- Training phase: Trains NFP NN.
- Validation phase: Validates NFP NN. Trains Only Classifier.
Each phase has two dataset:
- Training dataset: For train Classifier. (3500 records by default)
- Test dataset: For test Classifier. (500 records by default)

"""

from __future__ import print_function

import os
import sys
from argparse import ArgumentParser

import numpy as np
import six
from chainer import serializers

import data
import fp
import nn

parser = ArgumentParser(description='Neural Fingerprints')
parser.add_argument('-a', '--aid', type=int, default=686978,
                    help='select assay ID on pubchem')
parser.add_argument('-s', '--seed', type=int, default=123456789,
                    help='random seed of selecting data')
parser.add_argument('-d', '--dimension', type=int, default=128,
                    help='dimension of fingerprint')
parser.add_argument('-f', '--feature', type=int, default=128,
                    help='dimension of feature for generating fingerprint')
parser.add_argument('-m', '--middle', type=int, default=256,
                    help='number of nodes of middle layer of NN')
parser.add_argument('-r', '--radius', type=int, default=3,
                    help='radius for fingerprint')
parser.add_argument('-b', '--batchsize', type=int, default=50,
                    help='batchsize of NN')
parser.add_argument('--epocht', type=int, default=15,
                    help='number of epoch of NN train')
parser.add_argument('--epochv', type=int, default=20,
                    help='number of epoch of NN validation')
parser.add_argument('-e', '--epoch', type=int, default=15,
                    help='number of epoch overall')
parser.add_argument('-x', '--data', type=int, default=3500,
                    help='number of data of NN train')
parser.add_argument('-y', '--datatest', type=int, default=500,
                    help='number of data of NN test')

args = parser.parse_args()
aid = args.aid
seed = args.seed
d = args.dimension
f = args.feature
m = args.middle
R = args.radius
batchsize = args.batchsize
n_train_epoch = args.epocht
n_val_epoch = args.epochv
n_epoch = args.epoch
N = args.data
N_test = args.datatest

print('conventional approach')
print('aid           : {}'.format(aid))
print('seed          : {}'.format(seed))
print('d             : {}'.format(d))
print('f             : {}'.format(f))
print('u             : {}'.format(u))
print('R             : {}'.format(R))
print('batchsize     : {}'.format(batchsize))
print('n_train_epoch : {}'.format(n_train_epoch))
print('n_val_epoch   : {}'.format(n_val_epoch))
print('n_epoch       : {}'.format(n_epoch))
print('N             : {}'.format(N))
print('N_test        : {}'.format(N_test))

# Load data
print('load data...')
sids, assays = data.load_assay(aid, (N + N_test) * 2, seed)
sids_train, sids_val = [[], []], [[], []]
for i in six.moves.range(2):
    def add_and_pop(dst, src, n):
        dst += src[-n:]
        del src[-n:]
    add_and_pop(sids_train[1], sids[i], N_test // 2)
    add_and_pop(sids_train[0], sids[i], N // 2)
    add_and_pop(sids_val[1], sids[i], N_test // 2)
    add_and_pop(sids_val[0], sids[i], N // 2)
y_train, y_val = [[], []], [[], []]
y_train[0] = np.array([assays[sid] for sid in sids_train[0]], dtype=np.int32)
y_train[1] = np.array([assays[sid] for sid in sids_train[1]], dtype=np.int32)
y_val[0] = np.array([assays[sid] for sid in sids_val[0]], dtype=np.int32)
y_val[1] = np.array([assays[sid] for sid in sids_val[1]], dtype=np.int32)
print('done.')

# Setup NN
net = nn.NN(d=d, batchsize=batchsize, n_train_epoch=n_train_epoch,
            n_val_epoch=n_val_epoch, n_units=m)

# Setup NFP generator
_nfp = fp.nfp(d, f, R)

# Learn
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    result = _nfp.update(sids_train, y_train, net, train=True)
    print('train acc = %f' % result)
    result = _nfp.update(sids_val, y_val, net, train=False)
    print('validation acc = %f' % result)
