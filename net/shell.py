import multiprocessing
import time
import os
import torch
import numpy as np
import random

def fun(config_file,train_file,test_file, output_dir1,dataset,seed,num):
    print("start")
    os.system(f'python CNN_framework.py {config_file} {train_file} {test_file} {output_dir1} {dataset} {seed} {num}')

if __name__ == '__main__':
    dataset='circle'# 269
    config_file = fr'../data/{dataset}/config.txt'
    train_file = fr'../data/{dataset}/parametertrain.txt'
    test_file = fr'../data/{dataset}/parametertest.txt'
    num = 400
    seed=1
    output_dir1 = f'{dataset}_seed_{num}_{seed}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1, dataset, seed,num), )
    p.start()
    seed = 2
    output_dir1 = f'{dataset}_seed_{num}_{seed}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1, dataset, seed, num), )
    p.start()
    seed = 3
    output_dir1 = f'{dataset}_seed_{num}_{seed}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1, dataset, seed, num), )
    p.start()
    seed = 4
    output_dir1 = f'{dataset}_seed_{num}_{seed}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1, dataset, seed, num), )
    p.start()
    seed = 5
    output_dir1 = f'{dataset}_seed_{num}_{seed}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1, dataset, seed, num), )
    p.start()
