from abc import ABC
from ast import Raise
from math import log2
from pyexpat import model
from re import S
from typing import Callable, List

from matplotlib import container

import numpy as np
import numpy.typing as npt
import pandas as pd


from setFTs import minmax
from setFTs.utils import get_indicator_set,int2indicator
from setFTs import transformations


class SetFunction(ABC): 
    """A parent class solely for inheritance purposes"""   
    def __call__(self, indicators, count_flag=True):
        """
            :param indicators: two dimensional np.array with one indicator vector per row
            :type indicators: np.array of type np.int32 or np.bool
            :param count_flag: a flag indicating whether to count set function evaluations
            :type count_flag: bool
            :return: a np.array of set function evaluations
            :rtype: float
        """
        pass
    
    def gains(self, n : int , S0,maximize = True):
        """ Helper function for greedy min/max. Finds element that will increase the set function value the most, if added to an input subset.

        :param n: groundset size
        :type n: int
        :param S0: indicator vector to be improved upon
        :type S0: np.array of type np.int32 or np.bool
        :return: integer index of element that produces the biggest gain if changed to 1 and the corresponding value gain
        :rtype: (np.array,float)
        """
       
        N = np.arange(n)
        if maximize:
            max_value = -np.inf
            max_el = -1
            for element in N[True^S0]:
                curr_indicator = S0.copy()
                curr_indicator[element] = True
                curr_value = self(curr_indicator, count_flag=False)[0]
                if curr_value > max_value:
                    max_value = curr_value
                    max_el = element
                elif curr_value == max_value:
                    if np.random.rand() > 0.5:
                        max_value = curr_value
                        max_el = element
            return max_el, max_value
        else:
            min_value = np.inf
            min_el = -1
            for element in N[S0]:
                curr_indicator = S0.copy()
                curr_indicator[element] = True
                curr_value = self(curr_indicator, count_flag=False)[0]
                if curr_value < min_value:
                    min_value = curr_value
                    min_el = element
                elif curr_value == min_value:
                    if np.random.rand() > 0.5:
                        min_value = curr_value
                        min_el = element
            return min_el, min_value

    def minimize_greedy(self, n : int, max_card : int, verbose=False, force_card=False):
        """ Greedy minimization algorithm for set functions does not guarantee that the optimal solution is found
        
        :param n: groundset size
        :type n: int
        :param max_card: upper limit of cardinality up to which the greedy algorithm should check
        :type max_card: int
        :param verbose: flag to enable to print gain information for each cardinality
        :type verbose: bool
        :param force_card: flag that forces the algorithm to continue until specified max_card is reached
        :type force_card: bool
        :return: an array indicator vector of booleans that minimizes the setfunction and the evaluated setfunction for that indicator
        :rtype: (np.array,float)
        """
        S0 = np.zeros(n, dtype=bool)
        for t in range(max_card):
            i, value = self.gains(n, S0,maximize=True)
            if verbose:
                print('gains: i=%d, value=%.4f'%(i, value))
                print(S0.astype(np.int32))
            if value > 0 or force_card:
                S0[i] = 1
            else:
                break
        return S0, self(S0, count_flag=False)[0]

    def maximize_greedy(self, n : int, max_card : int, verbose=False, force_card=False):
        """ Greedy maximization algorithm for set functions. (Does not guarantee that the optimal solution is found)
        
        :param n: groundset size
        :type n: int
        :param max_card: upper limit of cardinality up to which the greedy algorithm should check
        :type max_card: int
        :param verbose: flag to enable to print gain information for each cardinality
        :type verbose: bool
        :param force_card: flag that forces the algorithm to continue until specified max_card is reached
        :type force_card: bool
        :return: an np.array indicator vector of booleans that maximizes the setfunction and the evaluated setfunction for that indicator
        :rtype: (np.array,float)
        """
        S0 = np.ones(n, dtype=bool)
        for t in range(n - max_card):
            i, value = self.gains(n, S0,maximize = False)
            if verbose:
                print('gains: i=%d, value=%.4f'%(i, value))
                print(S0.astype(np.int32))
            if value > 0 or force_card:
                S0[i] = 0
            else:
                break
        return S0, self(S0, count_flag=False)[0]
    
class WrapSignal(SetFunction):
    """Wrapper class for instantiating set functions with a full list of set function evaluations"""
    def __init__(self,signal : List[float]):
        """initializes setfunction with list

        :param signal: complete list of float outputs of a setfunction in lexicographical ordering
        :type signal: List[float]
        """
        self.n = int(log2(len(signal)))
        self.freqs = get_indicator_set(int(log2(len(signal))))
        self.coefs = signal
        self.call_counter = 0

    def __call__(self,indicator : npt.NDArray, count_flag : bool=True) -> npt.NDArray[np.float64]:
        """calls set function with np.array of indicator vectors

            :param indicators: two dimensional np.array with one indicator vector per row
            :type indicators: np.array of type np.int32 or np.bool
            :param count_flag: a flag indicating whether to count set function evaluations
            :type count_flag: bool
            :return: a np.array of set function evaluations
            :rtype: npt.NDArray[float64]
        """

        if len(indicator.shape) < 2:
            indicator = indicator[np.newaxis, :]
        
        power_of_twos = np.asarray(2**np.arange(self.n))
        result = []
        for ind in indicator:
            result += [self.coefs[int(power_of_twos.dot(ind))]]
        return np.asarray(result)

    def full_signal(self):
        sig = self.coefs
        return sig

    def spectral_energy(self,max_card,flag_rescale = True):
        """calculates the normalized coefficients per cardinality

        :param max_card: maximum cardinality for which to calculate the spectral energy
        :type max_card: int
        :param flag_rescale: flag indicating whether to average over all coefficients
        :type flag_rescale: int
        :returns: normalized coefficient of length max_card
        :rtype: List[float]
        """
        freqs = self.freqs
        coefs = self.coefs
        cardinalites = freqs.sum(axis =1)
        metric = lambda x: np.linalg.norm(x)**2
        avg_values = []
        for i in range(max_card+1):
            card_i = cardinalites == np.full(freqs.shape[0],i)
            if(flag_rescale):
                avg = metric(card_i*coefs)/metric(coefs)
                avg_values +=  [avg]
            else: 
                avg = metric(card_i*coefs)
                avg_values += [avg]
        return avg_values

    def transform_fast(self,model = '3',flag_normalization = True):
        """fast Fourier transformation algorithm
        
        :param model: basis upon which to calculate the transform see arxiv.org/pdf/2001.10290.pdf for more info
        :type model: str
        :param flag_normalization: boolean flag indicationg whether normalization for WHT (model 5) should be included in Fourier transform
        :type flag_normalization: bool
        :returns: sparseDSFTFunction object of the desired model 
        :rtype: sparseDSFTFunction
        """
        fast_transformer = transformations.FastSFT(model = model,flag_normalization=flag_normalization)
        return fast_transformer.transform(self.coefs)

    def transform_sparse(self,model = '3',k_max = None,eps = 1e-8,flag_print = False,flag_general = True):
        """sparse Fourier transformation algorithm
        
        :param model: basis upon which to calculate the Fourier transform
        :type model: str
        :param k_max: max number of coefficients to keep track of during computation
        :type k_max: int
        :param eps: eps: abs(x) < eps is treated as zero
        :type eps: float of form 1e-i
        :param flag_print: enables verbose mode for more information
        :type flag_print: bool
        :param flag_general: enables random one-hop Filtering
        :type flag_general: bool
        :returns: a sparseDSFTFunction object of the desired model 
        :rtype: sparseDSFTFunction
        """
        sparse_transformer = transformations.SparseSFT(self.n,model=model,k_max = k_max,eps = eps,flag_print= flag_print,flag_general=flag_general)
        return sparse_transformer.transform(self)

    def min(self):
        """ finds the subset that returns the smallest set function value
        
        :returns: indicator vector that minimizes the set function
        :rtype: npt.NDArray[bool] 
        """
        sig = self.coefs
        min_index = sig.index(min(sig))
        return int2indicator(min_index,self.n)
    
    def max(self):
        """ finds the subset that returns the largest set function value
        
        :returns: indicator vector that maximizes the set function
        :rtype: npt.NDArray[bool] 
        """
        sig = self.coefs
        max_index = sig.index(max(sig))
        return int2indicator(max_index,self.n)

    def export_to_csv(self,name ="sf"):
        """ exports the frequencies and coefficients into a csv file 
            
            :param name: name of the newly created file ending in .csv
            :param type: str
        """
        df_freqs = pd.DataFrame(self.freqs)
        df_coefs = pd.DataFrame(self.coefs)
        df_sf = pd.concat([df_freqs,df_coefs],axis =1)
        df_sf.to_csv(name +'.csv')
    
class WrapSetFunction(SetFunction):
    """Wrapper class for instantiating set functions with a callable function"""
    def __init__(self, s : Callable[[npt.NDArray],float],n, use_call_dict=False, use_loop=True):
        """init function

        :param s: callable function that takes a one-dimensional np.array and returns a float
        :type s: npt.NDArray[bool] - > float
        :param use_call_dict: flag indicating whether to use a dictionary for calling the setfunction
        which results in removing duplicate indicators
        :type use_call_dict: bool
        :param use_loop: flag indicating whether to use a loop for calling the setfunction  
        :type use_loop: bool  
        """
        self.s = s
        self.n = n
        self.call_counter = 0
        self.use_loop = use_loop
        if use_call_dict:
            self.call_dict = {}
        else:
            self.call_dict = None
        
    def __call__(self, indicator : npt.NDArray, count_flag=True) -> npt.NDArray[np.float64]:
        """calls set function with np.array of indicator vectors

            :param indicators: two dimensional np.array with one indicator vector per row
            :type indicators: np.array of type np.int32 or np.bool
            :param count_flag: a flag indicating whether to count set function evaluations
            :type count_flag: bool
            :return: a np.array of set function evaluations
            :rtype: npt.NDArray[float64]
        """
        
        if len(indicator.shape) < 2:
            indicator = indicator[np.newaxis, :]
        
        result = []
        if self.call_dict is not None:
            for ind in indicator:
                key = tuple(ind.tolist())
                if key not in self.call_dict:
                    self.call_dict[key] = self.s(ind)
                    if count_flag:
                        self.call_counter += 1
                result += [self.call_dict[key]]
            return np.asarray(result)
        elif self.use_loop:
            result = []
            for ind in indicator:
                result += [self.s(ind)]
                if count_flag:
                    self.call_counter += 1
            return np.asarray(result)
        else:
            if count_flag:
                self.call_counter += indicator.shape[0]
            return self.s(indicator)
        
    def transform_fast(self,model = '3',flag_normalization = True):
        """fast Fourier transformation algorithm (not advised)

        :param model: basis upon which to calculate the transform see arxiv.org/pdf/2001.10290.pdf for more info
        :type model: str
        :returns: a sparseDSFTFunction object of the desired model 
        :rtype: sparseDSFTFunction
        """
        print("Warning: About to execute 2^n queries of the setfunction. Not advised to be used with larger n. \n Do you wish to proceed? [y/n]")
        inp = input()
        if (inp == 'y'):
            inds = get_indicator_set(self.n)
            coefs = self(inds)
        elif (inp == 'n'):
            return
        else:
            print("Please input y to confirm or n to abort")
        fast_transformer = transformations.FastSFT(model = model,flag_normalization= flag_normalization)
        return fast_transformer.transform(coefs)
        
    def transform_sparse(self,model = '3',k_max = None,eps = 1e-8,flag_print = False,flag_general = True):
        """sparse Fourier transformation algorithm
        
        :param model: basis upon which to calculate the Fourier transform
        :type model: str
        :returns: a sparseDSFTFunction object of the desired model 
        :rtype: sparseDSFTFunction
        """
        sparse_transformer = transformations.SparseSFT(self.n,model=model, k_max = k_max,eps = eps,flag_print=flag_print,flag_general = flag_general)
        return sparse_transformer.transform(self)

class SparseDSFTFunction(SetFunction):

    def __init__(self, frequencies : npt.NDArray, coefficients : npt.NDArray[np.float64], model : str ,normalization_flag=False):
        """ Initializes a Sparse Fourier transformed set function object for the desired model

            :param frequencies: two dimensional np.array of type np.int32 or np.bool with one indicator vector per row
            :type frequencies: npt.NDArray
            :param coefficients: one dimensional np.array of corresponding Fourier coeffients
            :type coefficients: npt.NDArray[np.float64]
            :param model: either '3',','W3','4' or '5'(or'WHT')
            :type model: str
            :param normalization_flag: Only used for model 5/WHT flag indicates whether call should normalize values or not. Default is False
            :type normalization_flag: bool
        """
        self.freq_sums = frequencies.sum(axis=1)
        self.freqs = frequencies
        self.coefs = coefficients.astype(np.longdouble)
        self.call_counter = 0
        self.n = frequencies.shape[1]
        if model not in ['3','W3','4','5','WHT']:
            raise Exception("unknown model type. Please choose valid model type")
        self.model = model
        self.normalization = normalization_flag

    def __call__(self, indicators, count_flag=True):
        """ reconstructs the original set function evaluation for a set of input indicator vectors from it's sparse transformation

            :param indicators: two dimensional np.array with one indicator vector per row
            :type indicators: np.array of type np.int32 or np.bool
            :param count_flag: a flag indicating whether to count set function evaluations
            :type count_flag: bool
            :return: a np.array of set function evaluations
            :rtype: npt.NDArray[float64]
        """
        ind = indicators
        freqs = self.freqs
        coefs = self.coefs
        fsum = self.freq_sums
        if len(ind.shape) < 2:
            ind = ind[np.newaxis, :]
        # if(self.model == '3SI'):
        #     coefs = (-1)**fsum * coefs
        #     active = freqs.dot(np.logical_not(ind).T)
        #     active = active == 0
        if(self.model == '3'):
            coefs = coefs
            active = freqs.dot(np.logical_not(ind).T)
            active = active == 0
        if(self.model == 'W3'):
            coefs = (np.sqrt(3))**(fsum) * coefs
            active = freqs.dot(np.logical_not(ind).T)
            active = (active == 0) * (0.5**ind.sum(axis=1))
        if(self.model == '4'):
            active = freqs.dot(ind.T)
            active = active == 0
        if(self.model == '5' or self.model == 'WHT'):
            n = freqs.shape[1]
            factor = 1
            if self.normalization:
                factor = (1/2)**n
            A_cap_B = freqs.dot(ind.T)
            res = factor*((-1)**A_cap_B * coefs[:, np.newaxis]).sum(axis=0)
            return res
        res = (active * coefs[:, np.newaxis]).sum(axis=0)
        return res

    def full_signal(self):
        print("Warning: About to execute 2^n queries of the setfunction. Not advised to be used with larger n. \n Do you wish to proceed? [y/n]")
        inp = input()
        if (inp == 'y'):
            inds = get_indicator_set(self.n)
            coefs = self(inds)
            return coefs
        elif (inp == 'n'):
            return
        else:
            print("Please input y to confirm or n to abort")

    def shapley_values(self):
        """
        Calculates the Shapley Values for all elements in the ground set

        :returns: an np.array the length of the groundset of shapley values 
        :rtype: npt.NDArray[float64]
        """
        freqs = self.freqs
        coefs = self.coefs
        n = self.n 
        card_b = freqs.sum(axis =1)
        if(self.model == 'W3'):
            raise Exception("Unimplemented Error: Shapley Values for Model W3 not implemented yet")
        mask = card_b != 0
        card_b = card_b[mask]
        freqs = freqs[mask]
        coefs = coefs[mask]
        # if(self.model =='3SI'):
        #     shapleys_of_bs = ((-1)**(card_b))*(1/(card_b))
        if(self.model == '3'):
            shapleys_of_bs = 1/(card_b)
        if(self.model == '4'):
            shapleys_of_bs = (-1/(card_b))
        if(self.model == '5'):
            if self.normalization:
                shapleys_of_bs =2**(-n)*(((-1)**card_b - 1)/(card_b))
            else:
                shapleys_of_bs =((-1)**card_b - 1)/(card_b)
        res = []
        for i in range(n):
            indicator = np.zeros(n, dtype=np.int32)
            indicator[i] = 1
            contains_i = freqs.dot(indicator)
            res += [np.sum(shapleys_of_bs*contains_i*coefs)]
        return np.asarray(res)

    def minimize_MIP(self,C=1000., cardinality_constraint = None):
        """ utilizes a Mixed Integer Program solver to minimize  a set function value

        :param C: parameter for the MIP, if 1000. does not work, try larger values (see https://arxiv.org/pdf/2009.10749.pdf)
        :type C: int
        :param cardinality_constraint: function that evaluates to true if the cardinality constraint is met. Takes an integer as an input and evaluates to a bool (e.g cardinality_constraint=lambda x: x == 3)
        :type cardinality_constraint: int -> bool
        :returns: bitvector with the smallest function value and associated function value
        :rtype: (npt.NDArray[bool],float)
        """
        est = self
        # if self.model == '3SI':
        #     new_coefs = (-1)**self.freq_sums*self.coefs
        #     est = SparseDSFTFunction(self.freqs,new_coefs,model = self.model) 
        #     minvec, minval = minmax.minmax_dsft3(est,C,cardinality_constraint,maximize = False)
        if self.model == '3' or self.model == 'W3':
            minvec, minval = minmax.minmax_dsft3(est,C,cardinality_constraint,maximize = False)
        if self.model == '4':
            minvec, minval = minmax.minmax_dsft4(est,C,cardinality_constraint,maximize = False)
        if self.model == '5' or self.model == 'WHT':
            minvec, minval = minmax.minmax_wht(self,cardinality_constraint,maximize=False) 
        return minvec, minval

    def maximize_MIP(self,C=1000., cardinality_constraint = None):
        """ utilizes a Mixed Integer Program solver to maximize  a set function value

        :param C: parameter for the MIP, if 1000. does not work, try larger values (see https://arxiv.org/pdf/2009.10749.pdf)
        :type C: int
        :param cardinality_constraint: function that evaluates to true if the cardinality constraint is met. Takes an integer as an input and evaluates to a bool (e.g cardinality_constraint=lambda x: x == 3)
        :type cardinality_constraint: int -> bool   
        :returns: bitvector with the largest function value and associated function value
        :rtype: (npt.NDArray[bool],float)
        """
        est = self
        # if self.model == '3SI':
        #     new_coefs = (-1)**self.freq_sums*self.coefs
        #     est = SparseDSFTFunction(self.freqs,new_coefs,model = self.model) 
        #     minvec, minval = minmax.minmax_dsft3(est,C,cardinality_constraint,maximize = True)
        if self.model == '3' or self.model == 'W3':
            minvec, minval = minmax.minmax_dsft3(est,C,cardinality_constraint,maximize = True)
        if self.model == '4':
            minvec, minval = minmax.minmax_dsft4(est,C,cardinality_constraint,maximize = True)
        if self.model == '5' or self.model == 'WHT':
            minvec, minval = minmax.minmax_wht(self,cardinality_constraint,maximize = True) 
        return minvec, minval
    
    def spectral_energy(self,max_card,flag_rescale = True):
        """ Calculates the spectral energy for each cardinality

            :param max_card: Maximum Cardinality to consider
            :type max_card: int
            :param flag_rescale: flag indicating whether spectral energy per cardinality
                                should be rescaled to be relative to the total energy
            :flag_rescale: bool
            :returns: spectral energy per cardinality
            :rtype: List[float]
        """
        freqs = self.freqs
        coefs = self.coefs
        cardinalites = freqs.sum(axis =1)
        metric = lambda x: np.linalg.norm(x)**2
        avg_values = []
        for i in range(max_card+1):
            card_i = cardinalites == np.full(freqs.shape[0],i)
            if(flag_rescale):
                avg = metric(card_i*coefs)/metric(coefs)
                avg_values +=  [avg]
            else: 
                avg = metric(card_i*coefs)
                avg_values += [avg]
        return avg_values

    def force_k_sparse(self,k):
        """ creates a k-sparse estimate that only keeps the k largest coefficients
            
            :param k: number of coefficients to keep
            :type k: int
            :returns: a sparseDSFTFunction object with only the k largest coefficients
            :rtype: sparseDSFTFunction
        """
        freqs = self.freqs
        coefs = self.coefs
        m = np.size(coefs)
        k_biggest_coefidx = np.argpartition(np.abs(coefs),range(m))[-k:]
        freqs_k = freqs[k_biggest_coefidx]
        coefs_k = coefs[k_biggest_coefidx]
        return SparseDSFTFunction(freqs_k,coefs_k,model = self.model)

    def export_to_csv(self,name ="dsft"):
        """ exports the frequencies and coefficients into a csv file 
            
            :param name: name of the newly created file
            :type name: str
        """
        df_freqs = pd.DataFrame(self.freqs)
        df_coefs = pd.DataFrame(self.coefs)
        df_dsft = pd.concat([df_freqs,df_coefs],axis =1)
        df_dsft.to_csv(name + '.csv')
    
class DSFT3OneHop(SetFunction):
    
    def __init__(self, n : int, weights, set_function : Callable[[npt.NDArray],float],model : str):
        """
        @param n: integer for groundset size
        @param weights: ???
        @param set_function: callable function that takes a one-dimensional np.array and returns a float
        @param model: a string that's either '3' or 'W3' signaling which model to use
        """
        self.n = n
        self.weights = weights
        self.s = set_function
        self.call_counter = 0
        if (model != '3' and model != 'W3'):
            raise Exception('model has to be a string that is either 3 or W3')
        self.model = model
    
    def __call__(self, indicators : npt.NDArray, count_flag=True ):
        """
        @param indicators: two dimensional np.array of type np.int32 or np.bool
        with one indicator vector per row
        @param count_flag: a flag indicating whether to count set function evaluations
        @returns: a np.array of set function evaluations
        """
        if len(indicators.shape) < 2:
            indicators = indicators[np.newaxis, :]
        
        s = self.s
        weights = self.weights
        res = []
        for ind in indicators:
            nc = np.sum(ind)
            if count_flag:
                self.call_counter += (nc + 1)
            mask = ind.astype(np.int32)==1
            ind_shifted = np.tile(ind, [nc, 1])
            ind_shifted[:, mask] = 1-np.eye(nc, dtype=ind.dtype)
            ind_one_hop = np.concatenate((ind[np.newaxis], ind_shifted), axis=0)
            weight_s0 = np.ones(1)*(1 + weights[True^mask].sum())
            active_weights = np.concatenate([weight_s0, weights[mask]])
            res += [(s(ind_one_hop)*active_weights).sum()]
        res = np.asarray(res)
        return res
    
    def convertCoefs(self, estimate):
        freqs = estimate.freqs
        coefs = estimate.coefs
        coefs_new = []
        freqs = freqs.astype(bool)
        for key, value in zip(freqs, coefs):
            coefs_new += [value/(1 + self.weights[True^key].sum())]
        freq_sums = freqs.sum(axis=1)
        new_coefs = (-1)**freq_sums*coefs_new
        return SparseDSFTFunction(freqs.astype(np.int32), np.asarray(new_coefs),self.model)

class DSFT4OneHop(SetFunction):
    
    def __init__(self, n, weights, set_function,model):
        self.n = n
        self.weights = weights
        self.s = set_function
        self.call_counter = 0
        self.model = model
    
    def __call__(self, indicators, count_flag=True, sample_optimal=True):
        if len(indicators.shape) < 2:
            indicators = indicators[np.newaxis, :]
        
        s = self.s
        weights = self.weights
        if sample_optimal:
            res = []
            for ind in indicators:
                nc = ind.shape[0]-np.sum(ind)
                if count_flag:
                    self.call_counter += (nc + 1)
                mask = ind.astype(np.int32)==0
                ind_shifted = np.tile(ind, [nc, 1])
                ind_shifted[:, mask] = np.eye(nc, dtype=ind.dtype)
                ind_one_hop = np.concatenate((ind[np.newaxis], ind_shifted), axis=0)
                weight_s0 = np.ones(1)*(1 + weights[True^mask].sum())
                active_weights = np.concatenate([weight_s0, weights[mask]])
                res += [(s(ind_one_hop)*active_weights).sum()]
            res = np.asarray(res)
        else:
            res = s(indicators)
            for i, weight in enumerate(weights):
                ind_shifted = indicators.copy()
                ind_shifted[:, i] = 1
                res += weight*s(ind_shifted)
            if count_flag:
                self.call_counter += (self.n + 1) * indicators.shape[0]
        return res
    
    def convertCoefs(self, estimate):
        freqs = estimate.freqs
        coefs = estimate.coefs
        coefs_new = []
        freqs = freqs.astype(bool)
        for key, value in zip(freqs, coefs):
            coefs_new += [value/(1 + self.weights[True^key].sum())]
        return SparseDSFTFunction(freqs.astype(np.int32), np.asarray(coefs_new),model = self.model)

class WHTOneHop(SetFunction):
    def __init__(self, n, weights, set_function,model ):
        self.n = n
        self.weights = weights
        self.s = set_function
        self.call_counter = 0
        self.model = model
    
    def __call__(self, indicators, count_flag=True):
        if len(indicators.shape) < 2:
            indicators = indicators[np.newaxis, :]
        if count_flag:
            self.call_counter += (self.n + 1) * indicators.shape[0]
        s = self.s
        weights = self.weights
        res = s(indicators)
        for i, weight in enumerate(weights):
            ind_shifted = indicators.copy()
            ind_shifted[:, i] = True^ind_shifted[:, i]
            res += weight*s(ind_shifted)
        return res
    
def eval_sf(gt : SetFunction, estimate : SetFunction, n : int, n_samples=1000, err_types=["rel"], custom_samples=None, p=0.5):
    """evaluation function for setfunction. Compares an estimation to the ground truth

        :param gt: a SetFunction representing the ground truth
        :param estimate: a SetFunction 
        :param n: the size of the ground set
        :param n_samples: number of random measurements for the evaluation
        :param err_types: List of strings that are mae or relative reconstruction error 
        :returns: error values
        :rtype: List[float]
    """
    if custom_samples is None:
        ind = np.random.binomial(1, p, (n_samples, n)).astype(np.bool)
    else:
        ind = custom_samples
    gt_vec = gt(ind, count_flag=False)
    est_vec = estimate(ind, count_flag=False)
    # print(f'ground truth: {gt_vec[:10]}')
    # print(f'estimate: {est_vec[:10]}')
    errors = []
    for err_type in err_types:
        if err_type=="mae":
            errors += [(np.linalg.norm(gt_vec - est_vec, 1)/n_samples)]
        elif err_type=="rel":
            errors += [(np.linalg.norm(gt_vec - est_vec)/np.linalg.norm(gt_vec))]
        elif err_type=="inf":
            errors += [np.linalg.norm(gt_vec - est_vec, ord=np.inf)]
        elif err_type=="res_quantiles":
            errors += [np.quantile(np.abs(gt_vec - est_vec), [0.25, 0.5, 0.75])]
        elif err_type=="quantiles":
            errors += [np.quantile(np.abs(gt_vec), [0.25, 0.5, 0.75])]
        elif err_type=="res":
            errors += [gt_vec - est_vec]
        elif err_type=="raw":
            errors += [gt_vec, est_vec]
        elif err_type=="R2":
            gt_mean = np.mean(gt_vec)
            errors += [1 - np.mean((est_vec - gt_vec)**2)/np.mean((gt_vec - gt_mean)**2)]
        else:
            raise NotImplementedError("Supported error types: mae, rel, inf, res_quantiles, quantiles")
    return errors

def createRandomSparse(n, k, constructor, 
                      rand_sets=lambda size: np.random.binomial(1, 0.2, size),
                      rand_vals=lambda k: (-0.5+np.random.rand(k))*20):
    """creates a random k-sparse set function

    :param n: size of the ground set
    :param k: desired sparsity
    :param constructor: a Fourier Sparse SetFunction constructor
    :param rand_sets: a random zero-one vector generator
    :param rand_vals: a random Fourier coefficient generator
    :returns: a fourier sparse set function, the actual sparsity
    """
    freq_coef_dict = dict()
    freqs = rand_sets((k, n))
    coefs = rand_vals(k)
    coefs[coefs == 0] = 1
    for freq, val in zip(freqs, coefs):
        freq_coef_dict[tuple(freq.tolist())] = val
    freq_coef_dict.pop(tuple(np.zeros(n, dtype=np.int32).tolist()), None)
    freqs = np.asarray(list(freq_coef_dict.keys())).astype(np.int32)
    coefs = np.asarray(list(freq_coef_dict.values())).astype(np.float64)
    k = len(freqs)
    return constructor(freqs, coefs)

def build_from_csv(path,model = None):
    '''loads a setfunction from a csv file
    
    :param path: path to csv file
    :type path: str
    :param model: DSFT model to build (None to build set function)
    :type model: str
    :returns: SparseDSFTFunction or WrapSignal
    '''
    df = pd.read_csv(path) 
    df_freqs = df.iloc[:,1:-1]
    df_coefs = df.iloc[:,-1:]
    freqs = df_freqs.to_numpy().astype(np.int32)
    coefs = df_coefs.to_numpy().flatten().tolist()
    if model != None:
        return SparseDSFTFunction(freqs,coefs,model = model)
    else:
        return WrapSignal(coefs)