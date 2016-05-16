#!/usr/bin/env python

from __future__ import print_function

import sys

import chainer.functions as F
import chainer.links as L
import numpy as np
import six
from chainer import Chain, ChainList, Variable, cuda, optimizers

import data

AtomIdMax = 100

class nfp(object):

    """NFP manager

    This class has the generator function of NFP and
    updator of NN for learning the generator of NFP.

    Args:
        d: Dimension of NFP.
        f: Dimension of the feature for generating NFP.
        R: Radius for generating NFP.
        gpu (boolean): GPU flag. If you want to use GPU, set it True.
    """
    def __init__(self, d, f, R, gpu):
        self.d = d
        self.f = f
        self.R = R
        self.gpu = gpu
        g = ChainList(*[L.Linear(1, f) for i in six.moves.range(AtomIdMax)])
        H = ChainList(*[L.Linear(f, f) for i in six.moves.range(R)])
        W = ChainList(*[L.Linear(f, d) for i in six.moves.range(R + 1)])
        self.optimizer = optimizers.Adam()
        self.model = Chain(H=H, W=W, g=g)
        if gpu:
            self.model.to_gpu(0)
        self.optimizer.setup(self.model)
        self.to = [[] for i in six.moves.range(2)]
        self.atom_sid = [[] for i in six.moves.range(2)]
        self.anum = [[] for i in six.moves.range(2)]

    def get_nfp(self, sids, train=True):
        """Generates NFP.

        Args:
            sids (int[]): List of substance IDs.
            train (boolean): Training flag. If you want to train
                    the NFP NN, set it True, otherwise False.

        Returns:
            fp: Dictionary of NFPs. Key is a substance ID.
        """

        d, f, R = self.d, self.f, self.R

        def add_var(x):
            if self.gpu:
                return Variable(cuda.to_gpu(x, 0), volatile=not train)
            else:
                return Variable(x, volatile=not train)

        if train:
            ti = 0
        else:
            ti = 1
        to = self.to[ti]
        atom_sid = self.atom_sid[ti]
        anum = self.anum[ti]
        if len(to) == 0:
            for sid in sids:
                mol = data.load_sdf(sid)
                atoms = mol.GetAtoms()
                n = len(atoms)
                base = len(to)
                to += [[] for i in six.moves.range(n)]
                atom_sid += [sid for i in six.moves.range(n)]
                anum += [1 for i in six.moves.range(n)]
                for atom in atoms:
                    anum[base + atom.GetIdx()] = atom.GetAtomicNum()
                    to[base + atom.GetIdx()] = [base + n_atom.GetIdx()
                                                for n_atom in atom.GetNeighbors()]
            for i in six.moves.range(len(to)):
                if len(to[i]) == 0:
                    to[i].append(i)

        V = len(atom_sid)
        vec = [[] for i in six.moves.range(R + 1)]
        fp = {}
        for l in six.moves.range(R + 1):
            vec[l] = [add_var(np.zeros([1, f], dtype='float32'))
                      for i in six.moves.range(V)]
        for sid in sids:
            fp[sid] = add_var(np.zeros([1, d], dtype='float32'))
        for i in six.moves.range(V):
            vec[0][i] += self.model.g[anum[i]](add_var(np.array([[1]],
                                                                dtype='float32')))
        p = [[] for i in six.moves.range(R)]
        for l in six.moves.range(R):
            p[l] = [to[i][np.random.randint(len(to[i]))]
                    for i in six.moves.range(V)]
            for i in six.moves.range(V):
                vec[l + 1][i] = F.tanh(self.model.H[l]
                                       (vec[l][i] + vec[l][p[l][i]]))

        tmp = [[] for i in six.moves.range(R + 1)]
        for l in six.moves.range(R + 1):
            for i in six.moves.range(V):
                tmp[l].append(F.softmax(self.model.W[l](vec[l][i])))
        for l in six.moves.range(R + 1):
            for i in six.moves.range(V):
                fp[atom_sid[i]] += tmp[l][i]

        return fp

    def update(self, sids, y, net, train=True):
        """Updates NFP NN.

        Args:
            sids (int[]): Substance ID.
            y (np.array(int32[])[2]): Activity data. y[0] is for
                    the training dataset and y[1] is for the test dataset.
            net (nn.NN): Classifier of QSAR.
            train (boolean): Training flag. If you want to train
                    the NFP NN, set it True, otherwise False.

        Returns:
            result (float): Overall accuracy on the test dataset.
        """
        
        self.model.zerograds()

        fps = self.get_nfp(sids[0] + sids[1], train)

        x_train = [fps[sid] for sid in sids[0]]
        x_test = [fps[sid] for sid in sids[1]]
        for x in x_train:
            x.volatile = 'off'
        for x in x_test:
            x.volatile = 'off'

        result = net.train(x_train, y[0], x_test, y[1], train, self.gpu)

        self.optimizer.update()
        return result
