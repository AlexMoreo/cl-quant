import numpy as np

def mae(prevs, prevs_hat):
    return Mean(AE, prevs, prevs_hat)

def mse(prevs, prevs_hat):
    return Mean(SE, prevs, prevs_hat)

def mkld(prevs, prevs_hat):
    return Mean(KLD, prevs, prevs_hat)

def mnkld(prevs, prevs_hat):
    return Mean(NKLD, prevs, prevs_hat)

def mrae(prevs, prevs_hat, test_set_size):
    eps=1 / (2 * test_set_size) # it was proposed in literature an eps = 1/(2*T), with T the size of the test set
    def RAEeps(p, p_hat):
        return RAE(p,p_hat,eps)
    return Mean(RAEeps, prevs, prevs_hat)

def Mean(error_metric, prevs, prevs_hat):
    n = len(prevs)
    assert n == len(prevs_hat), 'wrong sizes'
    return np.mean([error_metric(prevs[i], prevs_hat[i]) for i in range(n)])

def AE(p, p_hat):
    return abs(p_hat-p)

def SE(p, p_hat):
    return (p_hat-p)**2

def KLD(p, p_hat, eps=1e-8):
    sp = p+eps
    sp_hat = p_hat + eps
    first = sp*np.log(sp/sp_hat)
    second = (1.-sp)*np.log(abs((1.-sp)/(1.-sp_hat)))
    return first + second

def NKLD(p, p_hat):
    ekld = np.exp(KLD(p, p_hat))
    return 2.*ekld/(1+ekld) - 1.

def RAE(p, p_hat, eps):
    return abs(p_hat-p+eps)/(p+eps)
