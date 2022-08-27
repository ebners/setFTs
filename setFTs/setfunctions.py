
class SetFunction(): 
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

        :param n:groundset size
        :type n: int
        :param S0: indicator vector to be improved upon
        :type S0: np.array of type np.int32 or np.bool
        :returns: integer index of element that produces the biggest gain if changed to 1 and the corresponding value gain
        :rtype: (np.array,float)
        """
        pass

    def minimize_greedy(self, n : int, max_card : int, verbose=False, force_card=False):
        """ Greedy minimization algorithm for set functions. (Does not guarantee that the optimal solution is found)
        :param n: groundset size
        :type n: int
        :param max_card: upper limit of cardinality up to which the greedy algorithm should check
        :type max_card: int
        :param verbose: flag to enable to print gain information for each cardinality
        :type verbose: bool
        :param force_card: flag that forces the algorithm to continue until specified max_card is reached
        :type force_card: bool
        :returns: an np.array indicator vector of booleans that minimizes the setfunction and the evaluated setfunction for that indicator
        :rtype:(np.array,float)
        """
        pass

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
        :returns: an np.array indicator vector of booleans that maximizes the setfunction and the evaluated setfunction for that indicator
        :rtype:(np.array,float)
        """
        pass
    
class WrapSignal(SetFunction):
    """Wrapper class for instantiating set functions with a full list of set function evaluations"""
    def __init__(self,signal):
        """initializes setfunction with list

        :param signal: complete list of float outputs of a setfunction in lexicographical ordering
        :type signal: List[float]
        """
        pass

    def __call__(self,indicator, count_flag =True) :
        """calls set function with np.array of indicator vectors

            :param indicators: two dimensional np.array with one indicator vector per row
            :type indicators: np.array of type np.int32 or np.bool
            :param count_flag: a flag indicating whether to count set function evaluations
            :type count_flag: bool
            :return: a np.array of set function evaluations
            :rtype: npt.NDArray[float64]
        """
        pass

    def spectral_energy(self,max_card,flag_rescale = True):
        """calculates the normalized coefficients per cardinality

        :param max_card: maximum cardinality for which to calculate the spectral energy
        :type max_card: int
        :param flag_rescale: flag indicating whether to average over all coefficients
        :type flag_rescale: int
        :returns: normalized coefficient of length max_card
        :rtype: List[float]
        """
        pass

    def transform_fast(self,model = '3'):
        """fast Fourier transformation algorithm
        :param model: basis upon which to calculate the transform see arxiv.org/pdf/2001.10290.pdf for more info
        :type model: str
        :returns: sparseDSFTFunction object of the desired model 
        :rtype: sparseDSFTFunction
        """
        pass

    def transform_sparse(self,model = '3',k_max = None,eps = 1e-8,flag_print = True,flag_general = True):
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
        pass

    def min(self):
        """ finds the subset that returns the smallest set function value
        :returns: indicator vector that minimizes the set function
        :rtype: npt.NDArray[bool] 
        """
        pass
    
    def max(self):
        """ finds the subset that returns the largest set function value
        :returns: indicator vector that maximizes the set function
        :rtype: npt.NDArray[bool] 
        """
        pass

    def export_to_csv(self,name ="sf.csv"):
        """ exports the frequencies and coefficients into a csv file 
            :param name: name of the newly created file ending in .csv
            :param type: str
        """
        pass
    
class WrapSetFunction(SetFunction):
    """Wrapper class for instantiating set functions with a callable function"""
    def __init__(self, s,n, use_call_dict=False, use_loop=True):
        """init function

        :param s: callable function that takes a one-dimensional np.array and returns a float
        :type s: npt.NDArray[bool] - > float
        :param use_call_dict: flag indicating whether to use a dictionary for calling the setfunction
        which results in removing duplicate indicators
        :type use_call_dict: bool
        :param use_loop: flag indicating whether to use a loop for calling the setfunction  
        :type use_loop: bool  
        """
        pass
        
    def __call__(self, indicator , count_flag=True):
        """calls set function with np.array of indicator vectors

            :param indicators: two dimensional np.array with one indicator vector per row
            :type indicators: np.array of type np.int32 or np.bool
            :param count_flag: a flag indicating whether to count set function evaluations
            :type count_flag: bool
            :return: a np.array of set function evaluations
            :rtype: npt.NDArray[float64]
        """
        pass

    def transform_fast(self,model = '3'):
        """fast Fourier transformation algorithm (not advised)

        :param model: basis upon which to calculate the transform see arxiv.org/pdf/2001.10290.pdf for more info
        :type model: str
        :returns: a sparseDSFTFunction object of the desired model 
        :rtype: sparseDSFTFunction
        """
        pass
        
    def transform_sparse(self,model = '3',k_max = None,eps = 1e-8,flag_print = True,flag_general = True):
        """sparse Fourier transformation algorithm
        :param model: basis upon which to calculate the Fourier transform
        :type model: str
        :returns: a sparseDSFTFunction object of the desired model 
        :rtype: sparseDSFTFunction
        """
        pass

class SparseDSFTFunction(SetFunction):

    def __init__(self, frequencies , coefficients , model : str ,normalization_flag=False):
        """ Initializes a Sparse Fourier transformed set function object for the desired model

            :param frequencies: two dimensional np.array of type np.int32 or np.bool with one indicator vector per row
            :type frequencies: npt.NDArray
            :param coefficients: one dimensional np.array of corresponding Fourier coeffients
            :type coefficients: npt.NDArray[np.float64]
            :param model: either '3','3SI','W3','4' or '5'(or'WHT')
            :type model: str
            :param normalization_flag: Only used for model 5/WHT flag indicates whether call should normalize values or not. Default is False
            :type normalization_flag: bool
        """
        pass

    def __call__(self, indicators, count_flag=True):
        """ reconstructs the original set function evaluation for a set of input indicator vectors from it's sparse transformation

            :param indicators: two dimensional np.array with one indicator vector per row
            :type indicators: np.array of type np.int32 or np.bool
            :param count_flag: a flag indicating whether to count set function evaluations
            :type count_flag: bool
            :return: a np.array of set function evaluations
            :rtype: npt.NDArray[float64]
        """
        pass

    def shapley_values(self):
        """
        Calculates the Shapley Values for all elements in the ground set

        :returns: an np.array the length of the groundset of shapley values 
        :rtype: npt.NDArray[float64]
        """
        pass

    def minimize_MIP(self,C=1000., cardinality_constraint = None):
        """ utilizes a Mixed Integer Program solver to minimize  a set function value

        :param C: parameter for the MIP, if 1000. does not work, try larger values (see https://arxiv.org/pdf/2009.10749.pdf)
        :type C: int
        :param cardinality_constraint: function that evaluates to true if the cardinality constraint is met. Takes an integer as an input and evaluates to a bool (e.g cardinality_constraint=lambda x: x == 3)
        :type cardinality_constraint: int -> bool
        :returns: bitvector with the smallest function value and associated function value
        :rtype: (npt.NDArray[bool],float)
        """
        pass

    def maximize_MIP(self,C=1000., cardinality_constraint = None):
        """ utilizes a Mixed Integer Program solver to maximize  a set function value

        :param C: parameter for the MIP, if 1000. does not work, try larger values (see https://arxiv.org/pdf/2009.10749.pdf)
        :type C: int
        :param cardinality_constraint: function that evaluates to true if the cardinality constraint is met. Takes an integer as an input and evaluates to a bool (e.g cardinality_constraint=lambda x: x == 3)
        :type cardinality_constraint: int -> bool   
        :returns: bitvector with the largest function value and associated function value
        :rtype: (npt.NDArray[bool],float)
        """
        pass
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
        pass

    def force_k_sparse(self,k):
        """ creates a k-sparse estimate that only keeps the k largest coefficients
            
            :param k: number of coefficients to keep
            :type k: int
            :returns: a sparseDSFTFunction object with only the k largest coefficients
            :rtype: sparseDSFTFunction
        """
        pass

    def export_to_csv(self,name ="dsft"):
        """ exports the frequencies and coefficients into a csv file 
            :param name: name of the newly created file
            :type name: str
        """
        pass
    
class DSFT3OneHop(SetFunction):
    
    def __init__(self, n : int, weights, set_function,model):
        """
        @param n: integer for groundset size
        @param weights: ???
        @param set_function: callable function that takes a one-dimensional np.array and returns a float
        @param model: a string that's either '3' or 'W3' signaling which model to use
        """
        pass
    
    def __call__(self, indicators, count_flag=True ):
        """
        @param indicators: two dimensional np.array of type np.int32 or np.bool
        with one indicator vector per row
        @param count_flag: a flag indicating whether to count set function evaluations
        @returns: a np.array of set function evaluations
        """
        pass
    
    def convertCoefs(self, estimate):
        pass

class DSFT4OneHop(SetFunction):
    
    def __init__(self, n, weights, set_function,model):
        pass
    
    def __call__(self, indicators, count_flag=True, sample_optimal=True):
        pass

    
    def convertCoefs(self, estimate):
        pass

class WHTOneHop(SetFunction):
    def __init__(self, n, weights, set_function,model ):
        pass
    
    def __call__(self, indicators, count_flag=True):
        pass
    
def eval_sf(gt : SetFunction, estimate : SetFunction, n : int, n_samples=1000, err_types=["rel"], custom_samples=None, p=0.5):
    """
        @param gt: a SetFunction representing the ground truth
        @param estimate: a SetFunction 
        @param n: the size of the ground set
        @param n_samples: number of random measurements for the evaluation
        @param err_type: either mae or relative reconstruction error
    """
    pass
def createRandomSparse(n, k, constructor, 
                      rand_sets,
                      rand_vals):
    """
    @param n: size of the ground set
    @param k: desired sparsity
    @param constructor: a Fourier Sparse SetFunction constructor
    @param rand_sets: a random zero-one vector generator
    @param rand_vals: a random Fourier coefficient generator
    @returns: a fourier sparse set function, the actual sparsity
    """
    pass

def build_from_csv(path,model = None):
    pass