#python module for loading datasets and creating set functions
import functools
import pathlib
import csv

import ck.kernel as ck

import numpy as np
from dsft import setfunctions



def sensor_information_gain(S, K, sigma=1):
    S = S.astype(np.bool)
    logdet = np.log(np.linalg.det(np.eye(S.sum()) + (1/sigma**2)*K[S][:, S]))
    return logdet


def sensorPlacement_berkeley():
    curr_path = pathlib.Path(__file__).parent.absolute()
    K = np.loadtxt(str(curr_path) + '/data/sensor_placement/Berkeley.csv', delimiter=',')
    n = K.shape[0]
    print(n)
    sf = functools.partial(sensor_information_gain, K=K)
    s  = setfunctions.WrapSetFunction(sf,n = n)
    return s

def sensorPlacement_rain():
    curr_path = pathlib.Path(__file__).parent.absolute()
    K = np.loadtxt(str(curr_path) + '/data/sensor_placement/Berkeley.csv', delimiter=',')
    n = K.shape[0]
    sf = functools.partial(sensor_information_gain, K=K)
    s  = setfunctions.WrapSetFunction(sf,n = n)
    return s

def sensorPlacement_california():
    curr_path = pathlib.Path(__file__).parent.absolute()
    K = np.loadtxt(str(curr_path) + '/data/sensor_placement/Berkeley.csv', delimiter=',')
    n = K.shape[0]
    sf = functools.partial(sensor_information_gain, K=K)
    s  = setfunctions.WrapSetFunciton(sf,n = n)
    return s


def load_bench_bitcount10():
    coefs=[]
    dir_path = pathlib.Path(__file__).parent.absolute()
    data_path = str(dir_path) + '/data/bitcount_10.csv'
    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            try:
                coefs += [float(line[10])]
            except:
                coefs = coefs
    return setfunctions.WrapSignal(coefs)

def load_bench(bench = 'susan_s'):
    """possible values for bench: bitcount10,susan_c10,susan_e10,susan_s10,bzip2d_10,bzip2e_10,jpeg_c10,jpeg_d10"""
    coefs=[]
    dir_path = pathlib.Path(__file__).parent.absolute()
    data_path = str(dir_path) + "/data/" + bench + ".csv"
    return setfunctions.build_from_csv(data_path)
