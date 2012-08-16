import numpy as np
from scipy import optimize
from statsmodels.base.model import _fit_mle_newton
from statsmodels.tools import add_constant

data= np.genfromtxt('/home/justin/allIV.csv', delimiter=',')
x = add_constant(data[:,0], prepend=1)
y = data[:,1].reshape(1000,1)
z = data[:,2:]

def estimate(beta):
    y1=np.copy(y)
    x1=np.copy(x)
    est_vect =  z * (y1 - np.dot(x1, beta))
    print 'here'
    nobs = est_vect.shape[0]
    eta_star = _modif_newton(np.zeros(est_vect.shape[1]), est_vect,
                            np.ones(1000) * (1. / nobs))
    denom = 1. + np.dot(eta_star, est_vect.T)
    new_weights = 1. / nobs * 1. / denom
    llr = np.sum(np.log(new_weights))
    return -1 * llr

def _log_star( eta1, est_vect, wts):
        """
        Parameters
        ---------
        eta1: float
            Lagrangian multiplier

        est_vect: nxk array
            Estimating equations vector

        wts: nx1 array
            observation weights

        Returns
        ------

        data_star: array
            The weighted logstar of the estimting equations

        Note
        ----

        This function is really only a placeholder for the _fit_mle_Newton.
        The function value is not used in optimization and the optimal value
        is disregarded when computng the log likelihood ratio.
        """
        nobs = est_vect.shape[0]
        data = est_vect.T
        data_star = np.log(wts).reshape(-1, 1)\
           + (np.sum(wts) + np.dot(eta1, data)).reshape(-1, 1)
        idx = data_star < 1. / nobs
        not_idx = ~idx
        data_star[idx] = np.log(1 / nobs) - 1.5 +\
                  2. * nobs * data_star[idx] -\
                  ((nobs * data_star[idx]) ** 2.) / 2
        data_star[not_idx] = np.log(data_star[not_idx])
        return data_star

def _hess( eta1, est_vect, wts):
        """
        Calculates the hessian of a weighted empirical likelihood
        provlem.

        Parameters
        ----------
        eta1: 1xm array.

        Value of lamba used to write the
        empirical likelihood probabilities in terms of the lagrangian
        multiplier.

        est_vect: nxk array
            Estimating equations vector

        wts: nx1 array
            observation weights

        Returns
        -------
        hess: m x m array
            Weighted hessian used in _wtd_modif_newton
        """
        nobs = est_vect.shape[0]
        data = est_vect.T
        wts = wts.reshape(-1, 1)
        data_star_doub_prime = np.copy(np.sum(wts) + np.dot(eta1, data))
        idx = data_star_doub_prime < 1. / nobs
        not_idx = ~idx
        data_star_doub_prime[idx] = - nobs ** 2
        data_star_doub_prime[not_idx] = - (data_star_doub_prime[not_idx]) ** -2
        data_star_doub_prime = data_star_doub_prime.reshape(nobs, 1)
        wtd_dsdp = wts * data_star_doub_prime
        return np.dot(data, wtd_dsdp * data.T)

def _grad(eta1, est_vect, wts):
        """
        Calculates the gradient of a weighted empirical likelihood
        problem.


        Parameters
        ----------
        eta1: 1xm array.

        Value of lamba used to write the
        empirical likelihood probabilities in terms of the lagrangian
        multiplier.

        est_vect: nxk array
            Estimating equations vector

        wts: nx1 array
            observation weights

        Returns
        -------
        gradient: m x 1 array
            The gradient used in _wtd_modif_newton
        """
        wts = wts.reshape(-1, 1)
        nobs = est_vect.shape[0]
        data = est_vect.T
        data_star_prime = (np.sum(wts) + np.dot(eta1, data))
        idx = data_star_prime < 1. / nobs
        not_idx = ~idx
        data_star_prime[idx] = 2. * nobs - (nobs) ** 2 * data_star_prime[idx]
        data_star_prime[not_idx] = 1. / data_star_prime[not_idx]
        data_star_prime = data_star_prime.reshape(nobs, 1)  # log*'
        return np.dot(data, wts * data_star_prime)

def _modif_newton(x0, est_vect, wts):
        """
        Modified Newton's method for maximizing the log* equation.

        Parameters
        ----------
        x0: 1x m array
            Iitial guess for the lagrangian multiplier

        est_vect: nxk array
            Estimating equations vector

        wts: nx1 array
            observation weights

        Returns
        -------
        params: 1xm array
            Lagragian multiplier that maximize the log-likelihood given
            `x0`.

        See Owen pg. 64
        """
        x0 = x0.reshape(est_vect.shape[1], 1)
        f = lambda x0: - np.sum(_log_star(x0.T, est_vect, wts))
        grad = lambda x0: - _grad(x0.T, est_vect, wts)
        hess = lambda x0: - _hess(x0.T, est_vect, wts)
        kwds = {'tol': 1e-8}
        res = _fit_mle_newton(f, grad, x0, (), kwds, hess=hess, maxiter=50, \
                              disp=0)
        return res[0].T
