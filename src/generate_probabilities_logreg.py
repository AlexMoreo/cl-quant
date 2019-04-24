from util.file import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.svm import SVC
from time import time



VECTORSPATH = '../vectors'

def __fit_predict_proba(Xtr, ytr, Xte, prob_estimator):
    prob_estimator.fit(Xtr, ytr)
    return prob_estimator.predict_proba(Xte)[:, 1]


def logistic_regression_fit_predict_proba(Xs, ys, Xt, nfolds = 10):
    print('Xtr=', Xs.shape, ys.mean())
    print('Xte=', Xt.shape)

    print('optimizing hyperparameters and training with all')
    param_grid = {'C': np.logspace(-5, 5, 11)}
    gridsearch = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, n_jobs=-1, cv=5, verbose=1, refit=True)

    tinit = time()
    gridsearch.fit(Xs, ys)
    lr_time = time() - tinit
    print('svm fit took {:.3f} seconds'.format(lr_time))
    print('best_params {}'.format(gridsearch.best_params_))
    logreg = gridsearch.best_estimator_

    logreg.fit(Xs,ys)
    yt_ = logreg.predict_proba(Xt)[:, 1]

    skf = StratifiedKFold(n_splits=nfolds)
    indexes = [(tr,te) for tr,te in skf.split(Xs, ys)]

    print('training folds')
    unordered = Parallel(n_jobs=-1)(delayed(__fit_predict_proba)(Xs[train_index], ys[train_index], Xs[test_index], clone(logreg)) for train_index, test_index in indexes)
    ys_ = np.zeros_like(ys).astype(np.float)
    for i,(_,test_index) in enumerate(indexes):
        ys_[test_index] = unordered[i]

    return ys_, yt_


for vectorset in list_dirs(VECTORSPATH):
    print('='*80)
    print(vectorset)
    vectorset=join(VECTORSPATH,vectorset)

    if exists(join(vectorset, 'train.y_prob.npy')) and exists(join(vectorset, 'test.y_prob.npy')): continue

    train_x = np.load(join(vectorset, 'train.vec.npy'))
    train_y = np.load(join(vectorset, 'train.y.npy'))
    test_x = np.load(join(vectorset, 'test.vec.npy'))
    test_y = np.load(join(vectorset, 'test.y.npy'))

    train_y_prob, test_y_prob = logistic_regression_fit_predict_proba(train_x, train_y, test_x)
    print('tr acc={:.3f}'.format(((train_y_prob > 0.5) == train_y).mean()))
    print('te acc={:.3f}'.format(((test_y_prob > 0.5) == test_y).mean()))

    np.save(join(vectorset, 'train.y_prob.npy'), train_y_prob)
    np.save(join(vectorset, 'test.y_prob.npy'), test_y_prob)