import multiprocessing
import time
import os
import random
def fun(config_file,train_file,test_file, output_dir1,dataset,seed,num):
    print("start")
    os.system(f'python ANN_framework.py {config_file} {train_file} {test_file} {output_dir1} {dataset} {seed} {num}')

# Artificial Neural Network-Based Compact Modeling Methodology for Advanced Transistors
if __name__ == '__main__':
    dataset = 'circle'
    config_file = fr'../data/{dataset}/config.txt'
    train_file = fr'../data/{dataset}/parametertrain.txt'
    test_file = fr'../data/{dataset}/parametertest.txt'
    # cmode=25
    num=400
    sd=2
    output_dir1 = f'{dataset}_seed_{num}_{sd}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file,train_file,test_file, output_dir1,dataset,sd,num), )
    p.start()
    sd=2
    output_dir1 = f'{dataset}_seed_{num}_{sd}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file,train_file,test_file, output_dir1,dataset,sd,num), )
    p.start()
    sd = 3
    output_dir1 = f'{dataset}_seed_{num}_{sd}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1, dataset, sd, num), )
    p.start()
    sd = 4
    output_dir1 = f'{dataset}_seed_{num}_{sd}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1, dataset, sd, num), )
    p.start()
    sd = 5
    output_dir1 = f'{dataset}_seed_{num}_{sd}'
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    p = multiprocessing.Process(target=fun(config_file, train_file, test_file, output_dir1, dataset, sd, num), )
    p.start()



