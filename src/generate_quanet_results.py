import torch
import numpy as np
from sklearn.svm import LinearSVC, SVC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.domain import pack_domains
from data.tasks import WebisCLS10_task_generator, WebisCLS10_crossdomain_crosslingual_task_generator
from domain_adaptation.dci import DCI
from domain_adaptation.pivotselection import pivot_selection
import os, sys
from time import time
from quantification.helpers import *
from quantification.metrics import *
from quantification.quanet import QuaNet
from util.plot_correction import plot_corr
from util.results import Result
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from os.path import join
import scipy
from util.file import *

cuda = torch.device('cuda')

VECTORSPATH = '../vectors'

sample_size = 200
samples_by_prevalence = 100

results_table = Result()

def add_results(results, dataset, method, approach):
    for metric,score in results.items():
        results_table.add(dataset=dataset, method=method, approach=approach, metric=metric, score=score)


for iter in range(10):
    hidden = 64
    drop_p = 0.0
    layers = 2
    ff = [1024, 512]
    config='QuaNet_drop{}_hidden{}_layers{}_ff{}'.format(drop_p, hidden, layers, '_'.join([str(x) for x in ff]))
    for vectorset_name in list_dirs(VECTORSPATH):
        print(vectorset_name)
        vectorset = join(VECTORSPATH, vectorset_name)

        train_x = np.load(join(vectorset, 'train.vec.npy'))
        train_y = np.load(join(vectorset, 'train.y.npy'))
        train_y_pred = np.load(join(vectorset, 'train.y_pred.npy'))
        train_y_prob = np.load(join(vectorset, 'train.y_prob.npy'))

        test_x = np.load(join(vectorset, 'test.vec.npy'))
        test_y = np.load(join(vectorset, 'test.y.npy'))
        test_y_pred = np.load(join(vectorset, 'test.y_pred.npy'))
        test_y_prob = np.load(join(vectorset, 'test.y_prob.npy'))

        tpr_val = tpr(train_y, train_y_pred)
        fpr_val = fpr(train_y, train_y_pred)
        ptpr_val = prob_tpr(train_y, train_y_prob)
        pfpr_val = prob_fpr(train_y, train_y_prob)

        input_size = train_x.shape[1]

        quanet = QuaNet(input_size, hidden, layers, ff, cuda, drop_p=drop_p)
        quanet.fit(train_x, train_y, train_y_pred, train_y_prob, sample_size=sample_size, lr=0.0001, wd=0, patience=20, validate_every=10, batch_size=21*20)

        print('Testing:')
        sample_indexes = sample_indexes_at_prevalence(test_y, prevalence_range(samples_by_prevalence), sample_size, seed=47)

        true_prevs, pred_prevs, cc_prevs, acc_prevs, pcc_prevs, apcc_prevs = QuaNet_predictions(quanet, test_x, test_y, test_y_pred, test_y_prob, tpr_val, fpr_val, ptpr_val, pfpr_val, sample_indexes)

        dataset, method = vectorset_name.split('.')

        results, err_vect = compute_errors(true_prevs, pred_prevs, sample_size)
        add_results(results, dataset, method, 'QuaNet')
        store_err_vectors(err_vect, 'QuaNet', vectorset)

        methods=[pred_prevs,cc_prevs,acc_prevs,pcc_prevs,apcc_prevs]
        labels=['QuaNet','CC','ACC','PCC','PACC']
        plot_corr(true_prevs, methods, labels, savedir='../plots', savename=vectorset_name+'_'+config+'.pdf', train_prev=None, test_prev=None, title='correction methods')

        results_table.dump('../results_'+config+'.csv')





