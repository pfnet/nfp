#!/usr/bin/env python

from __future__ import print_function

import sys

import chainer.functions as F
import chainer.links as L
import numpy as np
import six
from chainer import Chain, ChainList, Variable, optimizers

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
    """
    def __init__(self, d, f, R):
        self.d = d
        self.f = f
        self.R = R
        g = ChainList(*[L.Linear(1, f) for i in six.moves.range(AtomIdMax)])

        H = ChainList(*[ChainList(*[L.Linear(f, f)
                                    for i in six.moves.range(R)])
                        for j in six.moves.range(5)])
        W = ChainList(*[L.Linear(f, d) for i in six.moves.range(R)])
        self.model = Chain(H=H, W=W, g=g)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def get_nfp(self, sid, train=True):
        """Generates NFP.

        Args:
            sid (int): Substance ID.
            train (boolean): Training flag. If you want to train
                    the NFP NN, set it True, otherwise False.

        Returns:
            fp: NFP.
        """

        d, f, R = self.d, self.f, self.R
        mol = data.load_sdf(sid)
        atoms = mol.GetAtoms()
        n = len(atoms)
        fp = Variable(np.zeros([1, d], dtype='float32'), volatile=not train)
        r = [[Variable(np.zeros([1, f], dtype='float32'), volatile=not train)
              for i in six.moves.range(n)] for j in six.moves.range(R + 1)]
        for atom in atoms:
            a = atom.GetIdx()
            anum = atom.GetAtomicNum()
            r[0][a] += self.model.g[anum](Variable(np.array([[1]],
                                                            dtype='float32'),
                                                   volatile=not train))
        for l in six.moves.range(R):
            v = [Variable(np.zeros([1, f], dtype='float32'),
                          volatile=not train)
                 for i in six.moves.range(n)]
            for atom in atoms:
                a = atom.GetIdx()
                v[a] += r[l][a]
                for n_atom in atom.GetNeighbors():
                    na = n_atom.GetIdx()
                    v[a] += r[l][na]
            for atom in atoms:
                a = atom.GetIdx()
                deg = atom.GetDegree()
                deg = min(5, max(1, deg))
                r[l + 1][a] = F.tanh(self.model.H[deg - 1][l](v[a]))
                i = F.softmax(self.model.W[l](r[l + 1][a]))
                fp += i
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

        def get_nfps(sids, train=True):
            print('generate fingerprints...')
            fps = {}
            for i, sid in enumerate(sids[0] + sids[1]):
                fps[sid] = self.get_nfp(sid, train)
            print('done.')
            return fps

        self.model.zerograds()

        fps = get_nfps(sids, train)
        x_train = [fps[sid] for sid in sids[0]]
        x_test = [fps[sid] for sid in sids[1]]
        for x in x_train:
            x.volatile = 'off'
        for x in x_test:
            x.volatile = 'off'

        result = net.train(x_train, y[0], x_test, y[1], train)
        self.optimizer.update()
        return result
