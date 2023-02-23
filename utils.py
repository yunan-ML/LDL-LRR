import numpy as np
from scipy.stats import rankdata
from functools import wraps

EPS = 1e-15

def _check_params_for_measures(metric_fn):
    @wraps(metric_fn)
    def _func(inputs, targets):
        assert inputs.shape == targets.shape
        if len(inputs.shape) == 1:
            inputs, targets = inputs[None,:], targets[None,:]
        assert np.allclose(inputs.sum(1), 1)
        assert np.allclose(targets.sum(1), 1)
        assert ((inputs >= 0) & (targets >= 0)).all()
        return metric_fn(inputs, targets)
    return _func

@_check_params_for_measures
def Intersec(inputs, targets):
    targets = targets.copy()
    mask = np.where(inputs < targets)
    targets[mask] = inputs[mask]
    return targets.sum(1).mean()

@_check_params_for_measures
def Cosine(inputs, targets):
    s = (targets * inputs).sum(1)
    m = np.linalg.norm(targets, ord=2, axis=1) * np.linalg.norm(inputs, ord=2, axis=1)
    return (s / m).mean()

@_check_params_for_measures
def Clark(inputs, targets):
    return (np.sqrt((np.power(inputs - targets + EPS, 2) / np.power(inputs + targets + EPS, 2)).sum(1))).mean()

@_check_params_for_measures
def Cheb(inputs, targets):
    return (np.max(np.abs(targets - inputs), 1)).mean()

@_check_params_for_measures
def Canber(inputs, targets):
    return (np.abs(targets - inputs + EPS) / (targets + inputs + EPS)).sum(1).mean()

@_check_params_for_measures
def KL(inputs, targets):
    return ( ( targets * (np.log(targets + EPS) - np.log(inputs + EPS)) ).sum(1) ).mean()

@_check_params_for_measures
def Rho(inputs, targets):
    rA, rB = rankdata(inputs, axis=1), rankdata(targets, axis=1)
    cov = ((rA - np.mean(rA, axis=1, keepdims=True)) * (rB - np.mean(rB, axis=1, keepdims=True))).mean(axis=1)
    std = np.std(rA, axis=1) * np.std(rB, axis=1)
    rho = cov / (std + EPS)
    return rho.mean()

def report(inputs, targets, ds=None, method=None, out_form='print'):
    '''
    `method`: The instantiation of an estimator
    `type`:
        'print': print pretty table
        'pandas': pandas Series
    '''
    assert out_form in ['print', 'pandas']
    che = Cheb(inputs, targets)
    cla = Clark(inputs, targets)
    can = Canber(inputs, targets)
    kld = KL(inputs, targets)
    cos = Cosine(inputs, targets)
    ins = Intersec(inputs, targets)
    rho = Rho(inputs, targets)
    
    scores = np.round(np.array([che, can, cla, kld, cos, ins, rho]), 3).tolist()
    showls = ['Cheb', 'Canber', 'Clark', 'KL', 'Cosine', 'Intersec', 'Rho']
    if method is not None:
        showls = ['method'] + showls
        try:
            scores = [method.__class__.__name__] + scores
        except:
            scores = [method.__class__] + scores
    if ds is not None:
        showls = ['dataset'] + showls
        scores = [ds] + scores
    if out_form == 'print':
        from prettytable import PrettyTable
        tb = PrettyTable()
        tb.field_names = showls
        tb.add_row(scores)
        print(tb)
    elif out_form == 'pandas':
        from pandas import Series
        res = Series(data=scores, index=showls)
        return res