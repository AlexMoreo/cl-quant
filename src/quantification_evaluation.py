import numpy as np
from util.file import *
from quantification.helpers import *
from util.plot_correction import plot_corr
from util.results import Result

VECTORSPATH = '../vectors'

sample_size = 200
samples_by_prevalence = 100

results_table = Result()

def add_results(results, dataset, method, approach):
    for metric,score in results.items():
        results_table.add(dataset=dataset, method=method, approach=approach, metric=metric, score=score)


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

    sample_indexes = sample_indexes_at_prevalence(test_y, prevalence_range(samples_by_prevalence), sample_size, seed=47)

    true_prevs, cc_prevs, acc_prevs, pcc_prevs, apcc_prevs = Baselines_predictions(test_y, test_y_pred, test_y_prob,
                                                                                   tpr_val, fpr_val, ptpr_val, pfpr_val,
                                                                                   sample_indexes)
    dataset,method = vectorset_name.split('.')

    print('Classify-and-count')
    results, err_vect = compute_errors(true_prevs, cc_prevs, sample_size)
    add_results(results, dataset, method, 'CC')
    store_err_vectors(err_vect, 'CC', vectorset)

    print('Adjusted classify-and-count')
    results, err_vect = compute_errors(true_prevs, acc_prevs, sample_size)
    add_results(results, dataset, method, 'ACC')
    store_err_vectors(err_vect, 'ACC', vectorset)

    print('Probabilistic classify-and-count')
    results, err_vect = compute_errors(true_prevs, pcc_prevs, sample_size)
    add_results(results, dataset, method, 'PCC')
    store_err_vectors(err_vect, 'PCC', vectorset)

    print('Adjusted probabilistic classify-and-count')
    results, err_vect = compute_errors(true_prevs, apcc_prevs, sample_size)
    add_results(results, dataset, method, 'PACC')
    store_err_vectors(err_vect, 'PACC', vectorset)

    methods = [cc_prevs, acc_prevs, pcc_prevs, apcc_prevs]
    labels = ['CC', 'ACC', 'PCC', 'PACC']
    plot_corr(true_prevs, methods, labels, savedir='../plots', savename=vectorset_name+'.pdf', train_prev=None, test_prev=None, title='correction methods')

results_table.dump('../results_vectors.csv')






