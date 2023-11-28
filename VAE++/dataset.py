import math
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse


def loadTargetData(args):
    file = args.path + '/' + args.dataset + '/' + args.transaction
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values("uid")
    usersNum, itemsNum = args.user_num + 1, args.item_num + 1
    targetDict = tp.groupby('uid')['iid'].apply(list).to_dict()

    rows, cols = tp['uid'], tp['iid']
    targetData = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64', shape=(usersNum, itemsNum))
    return targetData, targetDict, usersNum, itemsNum


def loadAuxiliaryData(args):
    file = args.path + '/' + args.dataset + '/' + args.examination
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values("uid")
    usersNum, itemsNum = args.user_num + 1, args.item_num + 1
    auxiliaryDict = tp.groupby('uid')['iid'].apply(list).to_dict()

    rows, cols = tp['uid'], tp['iid']
    auxiliaryData = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64', shape=(usersNum, itemsNum))
    return auxiliaryData, auxiliaryDict


def loadTestData(args):
    file = args.path + '/' + args.dataset + '/' + args.test
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values("uid")
    testDict = tp.groupby('uid')['iid'].apply(list).to_dict()
    return testDict
