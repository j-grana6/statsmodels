"""
This script estimates and conducts inference on linear instrumental variable
models via empirical likelihood.

Given a supplied vector of endogenous, exogenous and instrumental variables,
the user can estimate the regression parameters, conduct hypthesis tests and
form confidence intervals.

General References:

Marsh, Mittlehammer and Judge "Empirical Likelihood Estimators of the
Linear Simultaneous Equation Model."  Selected Papers, American Agricultural
Economic Association.  Chicago, August 2001.

Qin and Lawless (1994).  "Empirical Likelihood and General Estimating Equations."
Annals of Statistics, 22, (1) 300-325.


"""


import numpy as np
from scipy import optimize
from descriptive2 import _OptFuncts
from statsmodels.tools import add_constant
from statsmodels.base.model import LikelihoodModel

data= np.genfromtxt('/home/justin/allIV.csv', delimiter=',')
x = add_constant(data[:,0], prepend=1)
y = data[:,1].reshape(1000,1)
z = data[:,2:]


class _IVOptimize(_OptFuncts):
    def __init__(self):
        pass

    def _est_eq_setup(self, beta, nuisance_param_num=None, params=None):
        instruments = self.instruments
        endog = self.endog
        exog=self.exog
        nobs = self.nobs
        if nuisance_param_num is not None:  # For Testing
            params[nusiance_param_num] = beta
        est_vect =  instruments *\
        (endog.reshape(nobs,1) - np.dot(exog, beta).reshape(nobs,1))

        eta_star = self._modif_newton(np.zeros(est_vect.shape[1]), est_vect,
            np.ones(nobs) * (1. / nobs))
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

    def fit(self):
        return super(ELIVRegress, self).fit(method='powell')

    def loglike(self, params):
        return self._est_eq_setup(params)

    def score(self, a):
        return None
