#!/usr/bin/env python
"""Data loader

Please edit the value of the variable 'base_path'.
The directory structure must be as below:

<base_path>/assay/<aid>.csv
<base_path>/substance/<sid / 10^6>/<sid / 10^3>/<sid>.sdf

"""

from __future__ import print_function

import gzip
import os
import re
import shutil

import numpy as np
import six
from rdkit import Chem

# Base path of data directory.
base_path = "/data0/t2g-15INH/pubchem"

def load_sdf(sid):
    substance_path = base_path + '/substance'
    sid_x = sid // (10 ** 6)
    sid_y = sid // (10 ** 3)
    element_file = '{}/{}/{}/{}.sdf'.format(substance_path, sid_x, sid_y, sid)
    try:
        mol = Chem.SDMolSupplier(element_file)[0]
    except Exception as e:
        print(e)
        print(element_file)
        return None
    else:
        return mol

def load_assay(aid, N, seed):
    assay_path = base_path + '/assay'
    assay_file = '%s/%d.csv' % (assay_path, aid)
    with open(assay_file) as f:
        recs = [[], []]
        for line in f:
            r = line.split(',')
            if r[0] == 'PUBCHEM_SID':
                continue
            sid = int(r[0])
            result = -1
            flag = 0
            if ',Active,' in line:
                flag |= 1
            if ',Inactive,' in line:
                flag |= 2
            if flag == 1:
                result = 1
            if flag == 2:
                result = 0
            if result != -1:
                recs[result].append(sid)
    N2 = N // 2
    res = []
    assays = {}
    np.random.seed(seed)
    for r in six.moves.range(2):
        perm = np.random.permutation(len(recs[r]))
        sids = []
        i = 0
        while len(sids) < N2:
            sid = recs[r][perm[i]]
            i += 1
            if load_sdf(sid) == None:
                continue
            sids.append(sid)
        for sid in sids:
            assays[sid] = r
        res.append(sids)
    return res, assays
