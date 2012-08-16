"""
This script estimates and conducts inference on linear instrumental variable
models via empirical likelihood.

Given a supplied vector of endogenous, exogenous and instrumental variables,
the user can estimate the regression parameters, conduct hypthesis tests and
form confidence intervals.

General References:

Marsh, Mittlehammer and Judge. "Empirical Likelihood Estimators of the
Linear Simultaneous Equation Model."  Selected Papers, American Agricultural
Economic Association.  Chicago, August 2001.

Qin and Lawless (1994)."Empirical Likelihood and General Estimating Equations."
Annals of Statistics, 22, (1) 300-325.


"""


import numpy as np
from scipy import optimize
from descriptive2 import _OptFuncts
from statsmodels.tools import add_constant
from statsmodels.base.model import LikelihoodModel
from scipy.stats import chi2

data = np.genfromtxt('/home/justin/allIV.csv', delimiter=',')
x = add_constant(data[:,0], prepend=1)
y = data[:,1].reshape(1000,1)
z = data[:,2:]


class _IVOptimize(_OptFuncts):
    def __init__(self):
        pass

    def _est_eq_setup(self, params, nuisance_param_idx=None, start_params=None):
        instruments = self.instruments
        endog = self.endog
        exog = self.exog
        nobs = self.nobs
        if nuisance_param_idx is not None:  # For Testing
            start_params[nuisance_param_idx] = params
            params = start_params
        est_vect = instruments *\
        (endog.reshape(nobs, 1) - np.dot(exog, params).reshape(nobs, 1))
        try:
            eta_star = self._modif_newton(np.zeros(est_vect.shape[1]), est_vect,
            np.ones(nobs) * (1. / nobs))
        except np.linalg.linalg.LinAlgError:
            return np.inf
        denom = 1. + np.dot(eta_star, est_vect.T)
        new_weights = 1. / nobs * 1. / denom
        loglik = np.sum(np.log(new_weights))
        return loglik


class ELIVRegress(_IVOptimize, LikelihoodModel):
    def __init__(self, endog, exog, instruments):
        self.nobs = exog.shape[0]
        self.endog = endog.reshape(self.nobs, 1)
        self.exog = exog
        self.instruments = instruments
        self.nexog = exog.shape[1]
        self.ninst = instruments.shape[1]

    def fit(self, start_params=None, method='powell', maxiter=100,
            full_output=True, disp=True, fargs=(), callback=None):
        res = super(ELIVRegress, self).fit(start_params=start_params,
                                        method=method, maxiter=maxiter,
                                        full_output=full_output, disp=disp,
                                        callback=callback)
        paramopt = res.params
        llf = - res.llf
        return ELIVResults(self, paramopt, llf)

    def loglike(self, params, nuis_param_idx=None, start_params = None):
        return self._est_eq_setup(params, nuisance_param_idx = nuis_param_idx,
                                  start_params= start_params)

    def score(self, a):
        return None


class ELIVResults(object):
    def __init__(self, model, paramopt, llmax):
        self.model = model
        self.params = paramopt
        self.llf = llmax

    def uncons_ll(self):
        return  np.sum(np.log(1. / self.model.nobs)) * self.model.nobs

    def spec_test(self):
        if self.model.ninst == self.model.nexog:
            raise Exception("There must be more instruments than exogenous\
                              variables to conduct a specification test")

        llr = - 2 * (self.llf - self.uncons_ll())
        pval = chi2.sf(llr, self.model.ninst - self.model.nexog)
        return llr, pval

    def test_beta(self, b0_vals, param_nums):
        start_params = np.copy(self.params)
        start_params[param_nums] = b0_vals
        nuis_idx = np.delete(np.arange(self.model.nexog), param_nums)
        f = lambda params: - self.model.loglike(params, nuis_param_idx=nuis_idx,
                                              start_params = start_params)
        x0 = start_params[nuis_idx]
        res = optimize.fmin_powell(f, x0, full_output=1)
        ll = res[1]
        llrstat = 2 * (ll - self.llf)
        return llrstat, chi2.sf(llrstat,1)
