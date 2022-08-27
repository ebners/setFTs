

def plot_freq_card(sf,plot_type = 'bar'):
    pass

def plot_freq_card_multi(sf_list,label_list,plot_type = 'bar'):
    pass

def plot_spectral_energy(sf,max_card,flag_rescale =True,plot_type = 'plot'):
    pass

def plot_spectral_energy_multi(sf_list,label_list,max_card,flag_rescale = True, plot_type = 'plot'):
    pass

def plot_scatter(sf,label,max_card):
    pass

def plot_max_greedy(sf_list,label_list,n,max_card):
    pass

def plot_min_greedy(sf_list,label_list,n,max_card):
    pass

def plot_max_mip(ft_list,label_list,max_card):
    pass

def plot_min_mip(ft_list,label_list,max_card):
    pass
    
def plot_reconstruction_error(sf,n,err_types = ['rel'],model = '3',flag_general = True):
    pass

def plot_reconstruction_error_biggest_coefs(sf,n,max_sparsity,interval,err_types = ['rel'],model ='3'):
    pass
        
def plot_minimization_found(sf,model = '3',greedy = False):
    pass
           
def plot_minimization_found_biggest_coefs(sf,max_sparsity,interval,model = '3',greedy = False):
    pass

