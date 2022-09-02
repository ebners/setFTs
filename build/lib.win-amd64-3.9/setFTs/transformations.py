from typing import List
from math import log2

import numpy as np
import scipy
import scipy.linalg

from setFTs import setfunctions as sf
from setFTs import utils 
from setFTs import fast 

class FastSFT:
    def __init__(self,model : str = '3',flag_normalization = True):
        """
        @param model: a string representing which model to use. 
        Can be either 3, 4 or WHT
        """
        self.model = model
        self.normalize = flag_normalization

    def transform(self,signal):
        """
        @param signal: complete list outputs of a setfunction in lexicographical ordering
        @returns a dsft.setfunction object of the chosen model
        """
        if(self.model == '3'):
            coefs = np.asarray(fast.fdsft3(signal))
            freqs = utils.get_indicator_set(int(log2(len(signal))))
            return sf.SparseDSFTFunction(freqs,coefs,'3')
        # elif(self.model == '3SI'):
        #     coefs = np.asarray(fast.fdsft3_selfInverse(signal))
        #     freqs = utils.get_indicator_set(int(log2(len(signal))))
        #     return sf.SparseDSFTFunction(freqs,coefs,'3SI')
        elif(self.model == '4'):
            coefs = np.asarray(fast.fdsft4(signal))
            freqs = utils.get_indicator_set(int(log2(len(signal))))
            return sf.SparseDSFTFunction(freqs,coefs,'4')
        elif(self.model == 'WHT' or self.model == '5'):
            coefs = np.asarray(fast.fwht(signal,self.normalize))
            freqs = utils.get_indicator_set(int(log2(len(signal))))
            return sf.SparseDSFTFunction(freqs,coefs,'5',normalization_flag = not self.normalize)
        
    def inverse(self,s_hat):
        """
        @param s_hat: a sparse setfunction representation (SparseDSFT3Function | SparseDSFT4Function | SparseWHTFunction)
        @returns : complete list of integer outputs of the original setfunction in lexicographical ordering
        """
        if(self.model == '3'):
            return fast.fidsft3(s_hat.coefs)
        if(self.model == '4'):
            return fast.fidsft4(s_hat.coefs)
        if(self.model == 'WHT' or self.model == '5'):
            return fast.fiwht(s_hat.coefs)
       
class SparseSFT:
    """
    @author: Enrico Brusoni
    """
    def __init__(self, n :int, eps=1e-8, flag_print=True, k_max=None, flag_general=True, model='3'):
        """
            @param n: ground set size
            @param eps: |x| < eps is treated as zero
            @param flag_print: printing flag
            @param k_max: the maximum amount of frequencies maintained in all steps but the last
            @param flag_general: this toggles the filtering by a random one hop filter and is 
            required to handle arbitrary/adversary Fourier coefficients.
            @param model: model for Fourier transform.
        """
        self.n = n
        self.k_max = k_max
        self.flag_print = flag_print
        self.eps = eps
        self.flag_general = flag_general
        if flag_general:
            self.weights = np.random.normal(0, 1, n)
        self.model = model

    def solve_subproblem(self, s, keys_old, coefs_old, measurements_previous, M_previous):
        n = self.n
        eps = self.eps
        model = self.model
        if self.k_max is None:
            keys_sorted = keys_old
        else:
            cards = keys_old.sum(axis=1)
            mags = -np.abs(coefs_old)
            criteria = np.zeros(len(keys_old), dtype=[('cards', '<i4'), ('coefs', '<f8')])
            criteria['cards'] = cards
            criteria['coefs'] = mags
            idx_order = np.argsort(criteria, order=('cards', 'coefs'))[:self.k_max]
            keys_sorted = keys_old[idx_order]

            M_previous = M_previous[idx_order][:, idx_order]
            measurements_previous = measurements_previous[idx_order]

        n1 = keys_sorted.shape[1]
        measurement_positions = np.zeros((keys_sorted.shape[0], n), dtype=np.int32)
        if (model == 'W3' or model == '3'):
            measurement_positions[:, :n1] = keys_sorted
            measurement_positions[:, n1+1:] = np.ones(n - n1-1, dtype=np.int32)
        if (model == '4'):
            measurement_positions[:, :n1] = 1 - keys_sorted
            measurement_positions[:, n1] = 1
        measurements_new = s(measurement_positions)
        if (model == 'W3'):
            rhs = np.concatenate([measurements_new[:, np.newaxis], 
                                 2./np.sqrt(3) * measurements_previous[:, np.newaxis] - 1./np.sqrt(3) * measurements_new[:, np.newaxis]],
                                axis=1)
        if (model == '3'):

            rhs = np.concatenate([measurements_new[:, np.newaxis],
                                 measurements_new[:, np.newaxis] - measurements_previous[:, np.newaxis]],
                                axis=1)
        if (model == '4'):
            rhs = np.concatenate([measurements_new[:, np.newaxis],
                                 measurements_previous[:, np.newaxis] - measurements_new[:, np.newaxis]],
                                axis=1)
        coefs = scipy.linalg.solve_triangular(M_previous, rhs, lower=True)
        n_queries = len(measurements_new)
        #support_first = support_second = np.where(np.abs(coefs[:,0]) + np.abs(coefs[:,1]) > 2*eps)[0]
        # support_first = np.where(np.abs(coefs[:, 0]) > eps)[0]
        #support_second = np.where(np.abs(coefs[:, 1]) > eps)[0]
        support_first = np.where(np.abs(coefs[:, 0]) > eps)[0]
        support_second = np.where(np.abs(coefs[:, 1]) > eps)[0]
        dim1 = len(support_first)
        dim2 = len(support_second)
        dim = len(support_first) + len(support_second)
        M = np.zeros((dim, dim), dtype=np.longfloat)
        if (model == 'W3'):
            M[:dim1, :dim1] = M_previous[support_first][:, support_first]
            M[dim1:, :dim1] = 0.5 * M_previous[support_second][:, support_first]
            M[dim1:, dim1:] = 0.5 * np.sqrt(3) * M_previous[support_second][:, support_second]
        if(model == '3'):
            M[:dim1, :dim1] = M_previous[support_first][:, support_first]
            M[dim1:, :dim1] = M_previous[support_second][:, support_first]
            M[dim1:, dim1:] = -M_previous[support_second][:, support_second]
        if(model == '4'):
            M[:dim1, :dim1] = M_previous[support_first][:, support_first]
            M[dim1:, :dim1] = M_previous[support_second][:, support_first]
            M[dim1:, dim1:] = M_previous[support_second][:, support_second]
        measurements = np.concatenate([measurements_new[support_first], measurements_previous[support_second]])
        if(model == 'W3' or model == '3'):
            keys_first = measurement_positions[support_first][:, :n1 + 1]
            keys_second = measurement_positions[support_second][:, :n1 + 1]
        if(model == '4'):
            keys_first = 1 - measurement_positions[support_first][:, :n1 + 1]
            keys_second = 1 - measurement_positions[support_second][:, :n1 + 1]
        keys_second[:, -1] = 1
        keys = np.concatenate([keys_first, keys_second], axis=0)
        fourier_coefs = np.concatenate([coefs[support_first][:, 0], coefs[support_second][:, 1]])
        return fourier_coefs, keys, measurements, M, n_queries

    def transform(self, X0):
        n = self.n
        model = self.model
        if self.flag_general:
            if model == '4':
                s = sf.DSFT4OneHop(n, self.weights, X0,model)
                self.hs = s
            if model == '3':
                s = sf.DSFT3OneHop(n, self.weights, X0,model)
                self.hs = s
            if model == 'W3':
                print('One-Hop Filtering for W3 not implemented. Proceding with unfiltered function')
                s = X0
        else:
            s = X0
        if(model == 'W3' or model == '3'):
            sN = s(np.ones(n, dtype=np.bool))[0]
        if(model == '4'):
            sN = s(np.zeros(n, dtype=np.bool))[0]
        M = np.ones((1, 1), dtype=np.float64)
        keys = np.zeros((1, 0), dtype=np.int32)
        fourier_coefs = np.ones(1, dtype=np.float64)*sN
        measurements = np.ones(1, dtype=np.float64)*sN
        partition_dict = {():sN}
            
        n_queries_total = 0
        for k in range(n):
            if len(list(partition_dict.keys())) == 0:
                keys = np.zeros((1, n), dtype=np.int32)
                fourier_coefs = np.zeros(1, dtype=np.float64)
                break
            try:
                fourier_coefs, keys, measurements, M, n_queries = self.solve_subproblem(s, keys, fourier_coefs, measurements, M)
            except ValueError as e:
                partition_dict = {}
            
            if self.flag_print:
                print('iteration %d: queries %d'%(k+1, n_queries))
            n_queries_total += n_queries

        if self.flag_print:
            print('total queries: %d'%n_queries_total)
        if(model == '3'):
            freq_sums = keys.sum(axis=1)
            new_coefs = (-1)**freq_sums*fourier_coefs
            estimate = sf.SparseDSFTFunction(keys, new_coefs, model = '3')
        if(model == 'W3'):
            estimate = sf.SparseDSFTFunction(keys, fourier_coefs, model = 'W3')
        if(model == '4'):
            estimate = sf.SparseDSFTFunction(keys, fourier_coefs, model = '4')

        if self.flag_general:
            if(model == 'W3'):
                return estimate
            else:
                estimate = s.convertCoefs(estimate) 
        return estimate