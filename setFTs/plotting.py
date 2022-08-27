from cProfile import label
from tempfile import NamedTemporaryFile

import numpy as np
import matplotlib.pyplot as plt

from dsft import setfunctions

def plot_freq_card(sf,plot_type = 'bar'):
    indicators = sf.freqs
    n = np.shape(indicators)[1]
    freq_count = np.zeros(n+1)
    cardinalites = indicators.sum(axis =1)
    for card in cardinalites:
        freq_count[card] += 1
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        if(plot_type == 'bar'):
            plt.bar(np.arange(0, n+1), freq_count)
        elif(plot_type == 'plot'):
            plt.plot(np.arange(0, n+1), freq_count)    
        plt.xlabel("Set cardinality")
        plt.ylabel("Frequency count")
        plt.show()
        plt.savefig(f.name,format='pdf')
        plt.close()

def plot_freq_card_multi(sf_list,label_list,plot_type = 'bar'):
    freq_count_list = []
    if(len(sf_list) > 1):
        width = 1/len(sf_list)
    else:
        width = 0.9
    offset = 0
    for sf in sf_list:
        indicators = sf.freqs
        n = np.shape(indicators)[1]
        freq_count = np.zeros(n+1)
        cardinalites = indicators.sum(axis =1)
        for card in cardinalites:
            freq_count[card] += 1
        freq_count_list += [freq_count]
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        for i in range(len(freq_count_list)):
            if(plot_type == 'bar'):
                plt.bar(np.arange(0, n+1) + offset,freq_count_list[i],width = width, label =label_list[i])
            elif(plot_type == 'plot'):
                plt.plot(np.arange(0, n+1), freq_count_list[i],label = label_list[i]) 
            offset += width 
        plt.legend()
        plt.xlabel("Set cardinality")
        plt.ylabel("Frequency count")
        plt.show()
        plt.savefig(f.name,format='pdf')
        plt.close()

def plot_spectral_energy(sf,max_card,flag_rescale =True,plot_type = 'plot'):
    n = np.shape(sf.freqs)[1]
    spec_energy = sf.spectral_energy(max_card,flag_rescale)
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        if(plot_type == 'bar'):
            plt.bar(np.arange(0, n+1), spec_energy)
        elif(plot_type == 'plot'):
            plt.plot(np.arange(0, n+1), spec_energy)   
        plt.bar(np.arange(0, n+1), spec_energy)
        plt.legend()
        plt.xlabel("Set cardinality")
        plt.ylabel("Spectral Energy")
        plt.show()
        plt.savefig(f.name,format='pdf')
        plt.close()

def plot_spectral_energy_multi(sf_list,label_list,max_card,flag_rescale = True, plot_type = 'plot'):
    spect_list =[]
    if(len(sf_list) > 1):
        width = 1/len(sf_list)
    else:
        width = 0.9
    offset = 0
    for sf in sf_list:
        spect_list += [sf.spectral_energy(max_card,flag_rescale)]
        n = np.shape(sf.freqs)[1]
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        for i in range(len(spect_list)):
            if(plot_type == 'bar'):
                plt.bar(np.arange(0, n+1) + offset, spect_list[i],width = width,label = label_list[i])
            elif(plot_type == 'plot'):
                plt.plot(np.arange(0, n+1), spect_list[i],label = label_list[i])
            offset += width
        plt.legend()
        plt.xlabel('Cardinality of Frequency')
        plt.ylabel('Avg. Coefficient')
        plt.xlim(0, max_card)
        plt.xticks(np.arange(0,max_card+1))
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()

def plot_scatter(sf,label,max_card):
    #metric = lambda x: np.linalg.norm(x)**2
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        #coef_size = metric(sf.coefs)
        cards = sf.freqs.sum(axis =1)
        plt.scatter(cards,sf.coefs,label = label)
        plt.xlabel('Cardinality of Frequency')
        plt.ylabel('Coefficient Value')
        plt.xlim(0, max_card)
        plt.xticks(np.arange(0,max_card+1))
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()

def plot_max_greedy(sf_list,label_list,n,max_card):
    values_list = []
    for sf in sf_list:
        values = []
        for card in range(0,max_card+1):
            _, value = sf.maximize_greedy(n,card)
            values += [value]
        values_list +=[values]
    
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        for i in range(len(sf_list)):
            plt.plot(values_list[i],label = label_list[i])
        plt.legend()
        plt.xlabel('cardinality constraint')
        plt.ylabel('maximal Value')
        plt.xlim(0, max_card)
        plt.xticks(np.arange(0,max_card+1))
        plt.ylim(bottom=0)
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()

def plot_min_greedy(sf_list,label_list,n,max_card):
    values_list = []
    for sf in sf_list:
        values = []
        for card in range(0,max_card+1):
            _, value = sf.minimize_greedy(n,card)
            values += [value]
        values_list +=[values]
    
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        for i in range(len(sf_list)):
            plt.plot(values_list[i],label = label_list[i])
        plt.legend()
        plt.xlabel('cardinality constraint')
        plt.ylabel('minimal Value')
        plt.xlim(0, max_card)
        plt.xticks(np.arange(0,max_card+1))
        plt.ylim(bottom=0)
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()

def plot_max_mip(ft_list,label_list,max_card):
    values_fts = []
    for ft in ft_list:
        values_ft = []
        for card in range(0,max_card+1):
            indicator,value =ft.maximize_MIP(cardinality_constraint = lambda x : x == card)
            values_ft += [value]
        values_fts += [values_ft]
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        for i in range(len(values_fts)):
            plt.plot(values_fts[i],label=label_list[i])
        plt.legend()
        plt.xlabel('cardinality constraint')
        plt.ylabel('maximal value')
        plt.xlim(0, max_card)
        plt.xticks(np.arange(0,max_card+1))
        plt.ylim(bottom=0)
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()

def plot_min_mip(ft_list,label_list,max_card):
    values_fts = []
    for ft in ft_list:
        values_ft = []
        for card in range(0,max_card+1):
            indicator,value =ft.maximize_MIP(cardinality_constraint = lambda x : x == card)
            values_ft += [value]
        values_fts += [values_ft]
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        for i in range(len(values_fts)):
            plt.plot(values_fts[i],label=label_list[i])
        plt.legend()
        plt.xlabel('cardinality constraint')
        plt.ylabel('minimal value')
        plt.xlim(0, max_card)
        plt.xticks(np.arange(0,max_card+1))
        plt.ylim(bottom=0)
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()
    
def plot_reconstruction_error(sf,n,err_types = ['rel'],model = '3',flag_general = True):
    num_errors = len(err_types)
    error_values = [[] for _ in range(num_errors)]
    for i in range(2,9):
        eps_i = float("1e-%d" %i)
        if model == '3':
            est = sf.transform_sparse(model = '3',eps = eps_i,flag_general = flag_general)
        if model == 'W3':
            est = sf.transform_sparse(model = 'W3',eps = eps_i,flag_general = flag_general)
        if model == '4':
            est = sf.transform_sparse(model = '4',eps = eps_i,flag_general = flag_general)
        
        errors = setfunctions.eval_sf(sf,est,n,n_samples = 10,err_types = err_types)
        for i in range(num_errors):
            error_values[i] += [errors[i]]
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        for i in range(num_errors):
            plt.subplot(1,num_errors,i+1)
            plt.plot(np.arange(2,9),error_values[i],label = err_types[i])
            plt.legend()
            plt.xlabel('eps =1e-i')
            plt.ylabel('error')
            plt.xlim(2, 8)
            #plt.xticks(np.arange(0,max_sparsity,interval))
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()

def plot_reconstruction_error_biggest_coefs(sf,n,max_sparsity,interval,err_types = ['rel'],model ='3'):
    num_errors = len(err_types)
    error_values = [[] for _ in range(num_errors)]
    if model == '3':
            est = sf.transform_sparse(model = '3')
    if model == 'W3':
            est = sf.transform_sparse(model = 'W3')
    if model == '4':
            est = sf.transform_sparse(model = '4')
    for i in range(1,max_sparsity,interval):
        est_i = est.force_k_sparse(i)
        errors = setfunctions.eval_sf(sf,est_i,n,n_samples = 100,err_types = err_types)
        for i in range(num_errors):
            error_values[i] += [errors[i]]
    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        for i in range(num_errors):
            plt.subplot(1,num_errors,i+1)
            plt.plot(np.arange(1,max_sparsity,interval),error_values[i],label = err_types[i])
            plt.legend()
            plt.xlabel('k-sparsity')
            plt.ylabel('error')
            plt.xlim(0, max_sparsity)
            #plt.xticks(np.arange(0,max_sparsity,interval))
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()
        
def plot_minimization_found(sf,model = '3',greedy = False):
    minvals = []
    for i in range(2,9):
        eps_i = float("1e-%d" %i)
        if model == '3':
            est = sf.transform_sparse(model = '3',eps =eps_i)
        if model == 'W3':
            est = sf.transform_sparse(model = 'W3',eps = eps_i)
        if model == '4':
            est = sf.transform_sparse(model = '4',eps = eps_i)
        if(greedy):
            minopt,_ = est.minimize_greedy()
        else:
            minopt, _ = est.minimize_MIP()
        minvals += sf(minopt).tolist()

    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        plt.plot(np.arange(2,9),minvals,label = '')
        plt.legend()
        plt.xlabel('eps =1e-i')
        plt.ylabel('minimization_value')
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()
        
def plot_minimization_found_biggest_coefs(sf,max_sparsity,interval,model = '3',greedy = False):
    minvals = []
    if model == '3':
        est = sf.transform_sparse(model = '3')
    if model == 'W3':
        est = sf.transform_sparse(model = 'W3')
    if model == '4':
        est = sf.transform_sparse(model = '4')
    for i in range(1,max_sparsity,interval):
        est_i = est.force_k_sparse(i)
        if(greedy):
            minopt,_ = est_i.minimize_greedy()
        else:
            minopt, _ = est_i.minimize_MIP()
        minvals += [sf(minopt)]

    with NamedTemporaryFile(suffix='.pdf',delete=False) as f:
        plt.plot(np.arange(1,max_sparsity,interval),minvals,label = '')
        plt.legend()
        plt.xlabel('k-max')
        plt.ylabel('minimization_value')
        plt.xlim(0, max_sparsity)
        #plt.xticks(np.arange(0,max_sparsity,interval))
        plt.savefig(f.name, format='pdf')
        plt.show()
        plt.close()

