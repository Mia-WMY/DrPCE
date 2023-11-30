import multiprocessing
import numpy as np
import time
import random

import os


def fun(config_file,train_file,test_file, output_dir1,poly,dataset,seed,num):
    # print("start")
    os.system(f'python pce_framework.py {config_file} {train_file} {test_file} {output_dir1} {poly} {dataset} {seed} {num}')

if __name__ == '__main__':
    dataset='circle'
    config_file = fr'../data/{dataset}/config.txt'
    train_file = fr'../data/{dataset}/parametertrain.txt'
    test_file = fr'../data/{dataset}/parametertest.txt'
    poly_degree = 3
    #[2,1357]
    #[3,3121]
    num = 400
    sd = 1
    output_dir1 = f'{dataset}{num}_seed_{sd}_{poly_degree}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1,poly_degree, dataset, sd, num), )
    p.start()
    sd = 2
    output_dir1 = f'{dataset}{num}_seed_{sd}_{poly_degree}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1,poly_degree, dataset, sd, num), )
    p.start()
    sd = 3
    output_dir1 = f'{dataset}{num}_seed_{sd}_{poly_degree}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1,poly_degree, dataset, sd, num), )
    p.start()
    sd = 4
    output_dir1 = f'{dataset}{num}_seed_{sd}_{poly_degree}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1,poly_degree, dataset, sd, num), )
    p.start()
    sd = 5
    output_dir1 = f'{dataset}{num}_seed_{sd}_{poly_degree}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1,poly_degree, dataset, sd, num), )
    p.start()

