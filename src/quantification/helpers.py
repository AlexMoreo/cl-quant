import numpy as np
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
from quantification.metrics import *
import scipy
from os.path import join

def tpr(y_true, y_hat):
    return y_hat[y_true==1].sum() / y_true.sum()

def fpr(y_true, y_hat):
    true_negs=(y_true==0)
    return y_hat[true_negs].sum() / true_negs.sum()

def prob_tpr(y, yhat):
    positives = y.sum()
    ptp = yhat[y==1].sum()
    return ptp / positives

def prob_fpr(y, yhat):
    negatives = (1-y).sum()
    pfp = yhat[y==0].sum()
    return pfp / negatives

def rebalance(X, y, new_balance):
    Xpos, Xneg = X[y==1], X[y==0]
    ndocs = len(y)
    curr_balance = y.mean()
    if curr_balance==new_balance:
        print("no undersampling")
    else:
        if curr_balance < new_balance:
            neg_prop = curr_balance * (1. / new_balance - 1)
            select_neg = int(ndocs * neg_prop)
            indexes = np.random.choice(Xneg.shape[0], select_neg, replace=False)
            Xneg = Xneg[indexes]
        elif curr_balance > new_balance:
            neg_balance = 1 - curr_balance
            pos_prop = neg_balance / (1. / new_balance - 1)
            select_pos = int(ndocs * pos_prop)
            indexes = np.random.choice(Xpos.shape[0], select_pos, replace=False)
            Xpos = Xpos[indexes]
        X = scipy.sparse.vstack((Xpos,Xneg))
        y = np.array([1]*Xpos.shape[0]+[0]*Xneg.shape[0])
        order = np.random.permutation(len(y))
        return X[order], y[order]
    return X,y


def choices(X, yhat, k):
    assert k>0, '0 choices requested'
    replace = True if k>X.shape[0] else False # cannot take more elements than the existing ones if replace=False
    indexes = np.random.choice(X.shape[0], k, replace=replace)
    return X[indexes], yhat[indexes]

def choices(X, yhat, k):
    assert k>0, '0 choices requested'
    replace = True if k>X.shape[0] else False # cannot take more elements than the existing ones if replace=False
    indexes = np.random.choice(X.shape[0], k, replace=replace)
    return X[indexes], yhat[indexes]

def classify_and_count(yhat):
    return (yhat > 0.5).sum() / len(yhat)

def probabilistic_classify_and_count(yhat):
    return yhat.sum() / len(yhat)

def adjusted_quantification(estim, tpr, fpr, clip=True):
    if (tpr - fpr) == 0:
        return -1
    adjusted = (estim - fpr) / (tpr - fpr)
    if clip:
        adjusted = max(min(adjusted, 1.), 0.)
    return adjusted


def prevalence_range(repeat=1):
    """
    :return: a vector of prevalences varying at steps of 0.05 where the fist and last elements (0 and 1) have been
    replaced by 0.01 and 0.99, i.e.: [0.01, 0.05, 0.10,..., 0.90, 0.95, 0.99]
    """
    prevs_range = np.arange(21) * 1 / 20
    prevs_range[0] += 0.01
    prevs_range[-1] -= 0.01
    if repeat>1:
        prevs_range=np.repeat(prevs_range,repeat)
    return prevs_range


def sample_indexes_at_prevalence(y, prevalences, sample_size, seed=None):

    if seed is not None:
        print('change seed')
        np.random.seed(seed)

    n_docs = len(y)
    indexes = np.arange(n_docs)
    positive_indexes = indexes[y == 1]
    negative_indexes = indexes[y == 0]
    n_pos = y.sum()
    n_neg = (1-y).sum()

    sample_indexes = np.zeros((len(prevalences), sample_size), dtype=np.int)

    for i,prevalence in enumerate(prevalences):
        ntake_pos = int(sample_size * prevalence)
        ntake_neg = sample_size - ntake_pos

        positive_sample = np.random.choice(positive_indexes, ntake_pos, replace=ntake_pos>n_pos)
        negative_sample = np.random.choice(negative_indexes, ntake_neg, replace=ntake_neg>n_neg)

        sample_indexes[i] = np.concatenate((positive_sample,negative_sample))

    return sample_indexes


def batch_from_indexes(X, y, yhat, yprob, sample_indexes, tpr_val, fpr_val, ptpr_val, pfpr_val):

    Xyhatyprob_samples, stats_samples, real_prevalences_samples = [], [], []
    for sample_indexes_i in sample_indexes:

        X_sample = X[sample_indexes_i]
        y_sample = y[sample_indexes_i]
        yhat_sample = yhat[sample_indexes_i]
        yprob_sample = yprob[sample_indexes_i]
        Xyhatyprob_sample = np.hstack((X_sample, yhat_sample.reshape(-1, 1), yprob_sample.reshape(-1, 1)))

        order = np.argsort(yprob_sample)
        Xyhatyprob_sample = Xyhatyprob_sample[order]

        cc = classify_and_count(yhat_sample)
        acc = adjusted_quantification(cc, tpr_val, fpr_val, clip=False)
        pcc = probabilistic_classify_and_count(yprob_sample)
        apcc = adjusted_quantification(pcc, ptpr_val, pfpr_val, clip=False)

        Xyhatyprob_samples.append(Xyhatyprob_sample)
        # stats_samples.append([cc, acc, pcc, apcc, tpr_val, fpr_val, ptpr_val, pfpr_val])
        stats_samples.append([cc, acc, pcc, apcc, tpr_val, fpr_val, ptpr_val, pfpr_val,
                              1-cc, 1-acc, 1-pcc, 1-apcc, 1-tpr_val, 1-fpr_val, 1-ptpr_val, 1-pfpr_val])
        real_prevalences_samples.append(y_sample.mean())

    Xyhatyprob_samples = np.asarray(Xyhatyprob_samples)
    stats_samples = np.asarray(stats_samples)
    real_prevalences_samples = np.asarray(real_prevalences_samples)

    return Xyhatyprob_samples, stats_samples, real_prevalences_samples


def baselines_from_indexes(y, yhat, yprob, sample_indexes, tpr_val, fpr_val, ptpr_val, pfpr_val):

    stats_samples, real_prevalences_samples = [], []
    for sample_indexes_i in sample_indexes:

        y_sample = y[sample_indexes_i]
        yhat_sample = yhat[sample_indexes_i]
        yprob_sample = yprob[sample_indexes_i]

        cc = classify_and_count(yhat_sample)
        acc = adjusted_quantification(cc, tpr_val, fpr_val, clip=True)
        pcc = probabilistic_classify_and_count(yprob_sample)
        apcc = adjusted_quantification(pcc, ptpr_val, pfpr_val, clip=True)

        stats_samples.append([cc, acc, pcc, apcc])
        real_prevalences_samples.append(y_sample.mean())

    stats_samples = np.asarray(stats_samples)
    real_prevalences_samples = np.asarray(real_prevalences_samples)

    return stats_samples, real_prevalences_samples


def sample_generator(X, y, yhat, yprob, prevalences, sample_size, seed=None):

    if seed is not None:
        print('change seed')
        np.random.seed(seed)

    tpr_val = tpr(y, yhat)
    fpr_val = fpr(y, yhat)
    ptpr_val = prob_tpr(y, yprob)
    pfpr_val = prob_fpr(y, yprob)

    while True:
        sample_indexes= sample_indexes_at_prevalence(y, prevalences, sample_size)
        yield batch_from_indexes(X, y, yhat, yprob, sample_indexes, tpr_val, fpr_val, ptpr_val, pfpr_val)


def QuaNet_predictions(quanet, X, y, yhat, yprob, tpr_val, fpr_val, ptpr_val, pfpr_val, sample_indexes, batch_size=21*5):
    true_prevs = []
    net_prevs = []
    stats = []

    nbatches = int(np.ceil(sample_indexes.shape[0] / batch_size))
    for b in range(nbatches):
        sample_indexes_batch = sample_indexes[b*batch_size:(b+1)*batch_size]
        batch_Xyhat, batch_stats, batch_real_prevalences = \
            batch_from_indexes(X, y, yhat, yprob, sample_indexes_batch, tpr_val, fpr_val, ptpr_val, pfpr_val)
        true_prevs.append(batch_real_prevalences)
        net_prevs.append(quanet.predict(batch_Xyhat, batch_stats))
        stats.append(batch_stats)

    true_prevs=np.concatenate(true_prevs)
    net_prevs=np.concatenate(net_prevs)
    stats = np.concatenate(stats)
    cc_prevs, acc_prevs, pcc_prevs, apcc_prevs = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    acc_prevs = np.clip(acc_prevs, 0, 1)
    apcc_prevs = np.clip(apcc_prevs, 0, 1)

    return true_prevs, net_prevs, cc_prevs, acc_prevs, pcc_prevs, apcc_prevs

def Baselines_predictions(y, yhat, yprob, tpr_val, fpr_val, ptpr_val, pfpr_val, sample_indexes, batch_size=21*5):
    true_prevs = []
    stats = []

    nbatches = int(np.ceil(sample_indexes.shape[0] / batch_size))
    for b in range(nbatches):
        sample_indexes_batch = sample_indexes[b*batch_size:(b+1)*batch_size]
        batch_stats, batch_real_prevalences = baselines_from_indexes(y, yhat, yprob, sample_indexes_batch, tpr_val, fpr_val, ptpr_val, pfpr_val)
        stats.append(batch_stats)
        true_prevs.append(batch_real_prevalences)

    true_prevs=np.concatenate(true_prevs)
    stats = np.concatenate(stats)
    cc_prevs, acc_prevs, pcc_prevs, apcc_prevs = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]

    return true_prevs, cc_prevs, acc_prevs, pcc_prevs, apcc_prevs


def eval(prevs, prevs_hat, sample_size, verbose=True, metrics=[mae, mse, mkld, mnkld]):
    results={}
    for metric in metrics:
        results[metric.__name__] = metric(prevs, prevs_hat)
        if verbose: print('{}={:.8f}'.format(metric.__name__, results[metric.__name__]))
    results[mrae.__name__] = mrae(prevs, prevs_hat, sample_size)
    if verbose: print('{}={:.8f}'.format(mrae.__name__, results[mrae.__name__]))
    return results



def store_err_vectors(err_vectors, method, vectorset_name):
    for key,vector in err_vectors.items():
        np.save(join(vectorset_name,'{}.{}.npy'.format(method,key)), vector)

def compute_errors(prevs, prevs_hat, sample_size):
    eps = 1 / (2 * sample_size)  # it was proposed in literature an eps = 1/(2*T), with T the size of the test set

    def RAEeps(p, p_hat):
        return (RAE(p, p_hat, eps)+RAE(1-p, 1-p_hat, eps))/2

    results = {}
    err_vectors = {}
    for metric in [AE, SE, KLD, NKLD, RAEeps]:
        err_vect = [metric(prevs[i], prevs_hat[i]) for i in range(len(prevs))]
        err_vectors[metric.__name__] = err_vect
        results[metric.__name__] = np.mean(err_vect)

    return results, err_vectors




