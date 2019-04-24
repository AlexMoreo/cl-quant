import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from time import time
from data.domain import pack_domains
from data.tasks import WebisCLS10_task_generator
from domain_adaptation.dci import DCI
from domain_adaptation.pivotselection import pivot_selection
import os, sys
from quantification.helpers import *
from util.file import create_if_not_exist
from os.path import join


dcf='cosine'
npivots = 450
dataset_home='../'
vectors='../vectors'

create_if_not_exist(vectors)

def __fit_predict(Xtr, ytr, Xte, svm):
    svm.fit(Xtr, ytr)
    return svm.predict(Xte)


def svm_fit_predict(Xs, ys, Xt, nfolds = 10):
    print('Xtr=', Xs.shape, ys.mean())
    print('Xte=', Xt.shape)

    parameters = {'C': [10 ** i for i in range(-5, 5)]}
    svm = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, verbose=1, cv=5, refit=True)

    tinit = time()
    svm.fit(Xs, ys)
    svm_time = time() - tinit
    print('svm fit took {:.3f} seconds'.format(svm_time))
    print('best_params {}'.format(svm.best_params_))
    svm = svm.best_estimator_

    # evaluation
    yt_ = svm.predict(Xt)

    skf = StratifiedKFold(n_splits=nfolds)
    indexes = [(tr,te) for tr,te in skf.split(Xs, ys)]

    print('training folds')
    unordered = Parallel(n_jobs=-1)(delayed(__fit_predict)(Xs[train_index], ys[train_index], Xs[test_index], clone(svm)) for train_index, test_index in indexes)
    ys_ = np.zeros_like(ys).astype(np.float)
    for i,(_,test_index) in enumerate(indexes):
        ys_[test_index] = unordered[i]

    return ys_, yt_


for source, target, oracle, taskname in WebisCLS10_task_generator(os.path.abspath(dataset_home)):

    print(taskname)
    outpath = join(vectors,'{}_{}_{}_{}.DCI_p{}_{}'.format(source.language, source.domain, target.language, target.domain, npivots, dcf))
    create_if_not_exist(outpath)

    # pivot selection
    s_pivots, t_pivots = pivot_selection(npivots, source.X, source.y, source.U, target.U, source.V, target.V,
                                         oracle=oracle, phi=30, show=min(10, npivots), cross=False)

    dci = DCI(dcf=dcf, unify=True, post='normal')
    dX, dU, dP, dV = pack_domains(source, target, s_pivots, t_pivots)
    dci.fit(dU, dP)
    dLatent = dci.transform(dX)
    Xs = dLatent[source.name()]
    Xt = dLatent[target.name()]

    ys_, yt_ = svm_fit_predict(Xs, source.y, Xt)

    acc = (target.y == yt_).mean()
    print('acc={:.3f}'.format(acc))

    np.save(join(outpath, 'train.vec.npy'), Xs)
    np.save(join(outpath, 'train.y.npy'), source.y)
    np.save(join(outpath, 'train.y_pred.npy'), ys_)

    np.save(join(outpath, 'test.vec.npy'), Xt)
    np.save(join(outpath, 'test.y.npy'), target.y)
    np.save(join(outpath, 'test.y_pred.npy'), yt_)


print('Done')