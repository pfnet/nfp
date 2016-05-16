#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six
from chainer import Variable, optimizers


class MLP(chainer.Chain):

    """Multi-layer perceptron for Classifier.

    Args:
        n_in (int): Dimension of the input layer.
        n_units (int): Dimension of the middle layer.
        n_out (int): Dimension of the output layer.
    """
    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class NN(object):

    """Classifier of QSAR.

    Args:
        d (int): Dimension of NFP.
        batchsize (int): Batchsize of the training dataset.
        n_train_epoch (int): The number of epoch of the training dataset of
                the training phase.
        n_val_epoch (int): The number of epoch of the training dataset of
                the validation phase.
        n_units (int): Dimension of the middle layer.
    """
    def __init__(self, d, batchsize, n_train_epoch, n_val_epoch, n_units):
        self.d = d
        self.batchsize = batchsize
        self.n_train_epoch = n_train_epoch
        self.n_val_epoch = n_val_epoch
        self.n_units = n_units
        self.model = L.Classifier(MLP(self.d, self.n_units, 2))
        self.model.o = optimizers.Adam()
        self.model.o.setup(self.model)

    def train(self, x_train, y_train, x_test, y_test, train):
        """Trains and tests the classifier of QSAR.

        Args:
            x_train (Variable): NFP for the training dataset.
            y_train (np.array(int32[])): Activity data
                    for the training dataset.
            x_test (Variable): NFP for the test dataset.
            y_test (np.array(int32[])): Activity data for the test dataset.
            train (boolean): Training flag. If you want to train
                    the *NFP NN*, set it True, otherwise False.

        Returns:
            result (float): Overall accuracy on the test dataset.
        """

        N = len(y_train)
        N_test = len(y_test)
        model = self.model

        # training
        if train:
            n_epoch = self.n_train_epoch
        else:
            n_epoch = self.n_val_epoch
        for epoch in six.moves.range(1, n_epoch + 1):
            perm = np.random.permutation(N)
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, N, self.batchsize):
                idx_max = min(N, i + self.batchsize)
                x = [x_train[perm[idx]]
                     for idx in six.moves.range(i, idx_max)]
                x = F.concat(x, axis=0)
                t = Variable(np.asarray(
                    y_train[perm[i:i + self.batchsize]]))

                model.o.update(model, x, t)

                sum_loss += float(model.loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            print('train mean loss=%f, accuracy=%f' %
                  (sum_loss / N, sum_accuracy / N))

        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        batchsize_test = N_test // 2
        for i in six.moves.range(0, N_test, batchsize_test):
            x = [x_test[idx] for idx in six.moves.range(
                i, min(N_test, i + batchsize_test))]
            x = F.concat(x, axis=0)
            x.volatile = 'off'
            t = Variable(np.asarray(y_test[i:i + batchsize_test]),
                         volatile='off')

            loss = model(x, t)
            sum_loss += loss
            sum_accuracy += float(model.accuracy.data) * len(t.data)

            print('batchtest mean loss=%f, accuracy=%f' %
                  (float(loss.data), float(model.accuracy.data)))
        sum_loss.backward()
        model.o.update()

        result = sum_accuracy / N_test
        return result
