# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:17:03 2022

@author: chengxx
"""
import random

import torch

'''frame work stage
#1. train model
#2. test model
#3. calculate error, generate report
#4. create gummel test figure
'''

import sys
import time
import numpy as np
import os
from tools.kop_calculator import KOP_calculator, diff_forward
import matplotlib.pyplot as plt
import shutil

pred_suffix = 'p.txt'
pred_r_suffix = 'pr.txt'
id_min = 1e-13
import re
from tools.seed import setup_seed
error_test_file = ['364', '449']



def model_pred_group(model, infile, outfile,dataset,seed):
    os.system('python cnn_test.py -tt=%s -model=%s -output=%s -set=%s -ds=%s' % (infile, model, outfile,dataset,seed))
    if not os.path.exists(outfile):
        raise FileNotFoundError(outfile)


# 1. train model
def get_model_and_time(train_file, outdir,dataset,num,seed):
    train_file = os.path.abspath(train_file)
    print(os.path.join(outdir, f'model{num}_{seed}.pt'))
    model_file = os.path.join(outdir, f'model{num}_{seed}.pt')
    os.system('python cnn_train.py -data=%s -model=%s -set=%s -num=%s -ds=%s' % (train_file, model_file,dataset,num,seed))
    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)
    return model_file


# 2. test model
def test_model(model, test_file, output_dir,dataset,seed):
    test_file = os.path.abspath(test_file)
    file_dir = os.path.dirname(test_file)
    with open(test_file) as fid:
        contents = fid.readlines()
        pname = [re.match('\w+', e).group() for e in contents[0].split(',')[:-1]]
        contents = contents[1:]
    for ln in contents:
        p = ln.strip().split(',')
        test_file_name = p[-1].strip()
        if test_file_name in error_test_file:
            continue
        fn = os.path.join(file_dir, '%s.txt' % test_file_name)
        pnr = [','.join(pname)]
        pnr.append(','.join(e for e in p[:-1]))
        test_one_file(model, fn, pnr, output_dir,dataset,seed)


def test_one_file(model, ftest, parameter, output_dir,dataset,seed):
    with open(ftest) as f:
        test_content = f.readlines()
    p_content = parameter[:]
    p_content.append('vgs,vds')
    p_content.extend([','.join(e.split('\t')[:2]) for e in test_content[1:]])
    pred_file = os.path.join(output_dir, '%s%s' % (os.path.splitext(os.path.split(ftest)[1])[0], pred_suffix))
    pred_rfile = os.path.join(output_dir, '%s%s' % (os.path.splitext(os.path.split(ftest)[1])[0], pred_r_suffix))
    with open(pred_file, 'w') as f:
        f.write('\n'.join(p_content))
    model_pred_group(model, pred_file, pred_rfile,dataset,seed)


# 3. calculate error
# calculate KOP for each parameter
# calculate KOP for each parameter
# calculate error for all data
# generate report.txt for average KOP
def calculate_error(config_file, test_file, output_dir):
    test_file = os.path.abspath(test_file)
    file_dir = os.path.dirname(test_file)
    with open(test_file) as f:
        contents = f.readlines()[1:]
    file_names = [e.strip().split(',')[-1].strip() for e in contents]
    file_names = [e for e in file_names if e not in error_test_file]
    ori_files = [os.path.join(file_dir, '%s.txt' % e) for e in file_names]
    pred_files = [os.path.join(output_dir, '%s%s' % (e, pred_r_suffix)) for e in file_names]
    vtgm_ms, vtgm_ss, idsat_ms, idsat_ss, ioff_ms, ioff_ss = [], [], [], [], [], []
    rmses = []
    for i in range(len(file_names)):
        ori_data = np.loadtxt(ori_files[i], dtype='float', delimiter='\t', skiprows=1)[:, 1:]
        ids_m = ori_data[:, 1].reshape(-1)
        valid_id = ids_m > id_min
        vds_m = ori_data[:, 0].reshape(-1)
        valid_vd = vds_m != 0.0
        valid = np.logical_and(valid_vd, valid_id)
        if np.sum(valid) <= 0:
            continue
        ids_m = np.log2(ids_m[valid])
        ids_s = np.loadtxt(pred_files[i], dtype='float', delimiter=',', skiprows=3)[:, 2][valid]
        ids_s = np.log2(np.abs(ids_s.reshape(-1)))
        rmses.append(np.sqrt(np.mean(np.power((ids_s - ids_m) / ids_m,2))))
        vtgm_m, idsat_m, _, _, ioff_m = KOP_calculator(config_file, ori_files[i]).get_kop()
        vtgm_s, idsat_s, _, _, ioff_s = KOP_calculator(config_file, pred_files[i]).get_kop()
        vtgm_ms.append(vtgm_m)
        vtgm_ss.append(vtgm_s)
        idsat_ms.append(idsat_m)
        idsat_ss.append(idsat_s)
        ioff_ms.append(ioff_m)
        ioff_ss.append(ioff_s)
    vtgm_m_np, vtgm_s_np = np.array(vtgm_ms), np.array(vtgm_ss)
    idsat_m_np, idsat_s_np = np.array(idsat_ms), np.array(idsat_ss)
    ioff_m_np, ioff_s_np = np.array(ioff_ms), np.array(ioff_ss)
    vtgm_error = np.abs(vtgm_m_np - vtgm_s_np)
    idsat_error = np.abs((idsat_m_np - idsat_s_np) / idsat_m_np)
    ioff_error = np.abs((ioff_m_np - ioff_s_np) / ioff_m_np)
    result = ['_,avg,min,max']
    result.append('vth,%s,%s,%s' % (np.mean(vtgm_error), np.min(vtgm_error), np.max(vtgm_error)))
    result.append('Idsat,%s,%s,%s' % (np.mean(idsat_error), np.min(idsat_error), np.max(idsat_error)))
    result.append('Ioff,%s,%s,%s' % (np.mean(ioff_error), np.min(ioff_error), np.max(ioff_error)))
    rmse_np = np.array(rmses)
    result.append('rmse,%s,%s,%s' % (np.mean(rmse_np), np.min(rmse_np), np.max(rmse_np)))
    return result


def write_report(result, train_time, pred_time, output_dir):
    result.append('train_time,%s' % train_time)
    result.append('pred_time,%s' % pred_time)
    print("train time:",train_time)
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write('\n'.join(result))


# 4. calculate gummel test
def calculate_gummel_test(model, config_file, output_dir,dataset,seed):
    # choose two device parameter set from config file. upper, lower
    config_file = os.path.abspath(config_file)
    with open(config_file) as f:
        config_content = [e.strip() for e in f.readlines()]
    vdd, vdlin, parameters, lb, ub = '', '', '', '', ''
    for line in config_content:
        if not line:
            continue
        values = re.split('\s+', line)[1]
        if 'vdd' in line:
            vdd = float(values)
        if 'vdlin' in line:
            vdlin = float(values)
        if 'parameter' in line:
            parameters = values
        if 'lower' in line:
            lb = values
        if 'upper' in line:
            ub = values
    p = [[parameters, lb], [parameters, ub]]
    vxs = np.linspace(-2 * vdlin, 2 * vdlin, 101)
    # get vg
    vg_all = [vdlin, 0.5 * (vdlin + vdd), vdd]
    for idx1, params in enumerate(p):
        if idx1 == 1:
            parame_name = 'lower'
        else:
            parame_name = 'upper'
        for idx2, vg in enumerate(vg_all):
            pred_file = os.path.join(output_dir, '%s%s%s' % (parame_name, vg, pred_suffix))
            predr_file = os.path.join(output_dir, '%s%s%s' % (parame_name, vg, pred_r_suffix))
            result = params[:]
            vgs = np.abs(vxs) + vg
            vds = vxs * 2
            result.append('vg,vd')
            for i, vg in enumerate(vgs):
                result.append('%f,%f' % (vg, vds[i]))
            with open(pred_file, 'w') as f:
                f.write('\n'.join(result))
            model_pred_group(model, pred_file, predr_file,dataset,seed)
            ids = np.loadtxt(predr_file, dtype='float', delimiter=',', skiprows=3)[:, 2].reshape(-1)
            # datas = {'Ids_Vx': np.c_[vxs, ids]}
            plot_figure('Parmeters(%s),Vg(%s),I' % (parame_name, idx2), 'Vx', 'I', '', {'Ids_Vx': np.c_[vxs, ids]},
                        output_dir)
            derivative = diff_forward(vxs, ids)
            vxs1 = (vxs[1:] + vxs[:-1]) / 2
            # datas['dIds/dVx_Vx'] = np.c_[vxs1, derivative]
            plot_figure('Parmeters(%s),Vg(%s),I\'' % (parame_name, idx2), 'Vx', 'I\'', '',
                        {'dIds/dVx_Vx': np.c_[vxs1, derivative]}, output_dir)
            vxs2 = (vxs1[1:] + vxs1[:-1]) / 2
            # datas['d^2Ids/d^2Vx_Vx'] = np.c_[vxs2, diff_forward(vxs1, derivative)]
            # plot_figure('Parmeters(%s),Vg(%s),I(all)' % (parame_name, idx2), 'Vx', 'I\'\'', '',
            #            datas, output_dir)
            plot_figure('Parmeters(%s),Vg(%s),I\'\'' % (parame_name, idx2), 'Vx', 'I\'\'', '',
                        {'d^2Ids/d^2Vx_Vx': np.c_[vxs2, diff_forward(vxs1, derivative)]}, output_dir)

    # calculate all Ids under given p, vg, vx
    # for p
    # for vg
    #    # for vx
    #    # vgs = vg+vx, vds = 2*vx
    #    # x = concat(p, vgs, vds)
    #    # Ids = test(model, x)
    #    # plot(vx, Ids)
    #    # plot(vx, dIds/dVx)
    #    # plot(vx, d^2Ids/d^2Vx)
    #    # save one figure


def plot_figure(title, x_label, y_label, legstr, datas, save_path):
    plt.figure(figsize=(16, 16))
    plt.title(title)
    legs = []
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for key, data in datas.items():
        legs.append(legstr + str(key))
        plt.plot(data[:, 0], data[:, 1])
    plt.legend(legs, loc='lower right')
    plt.savefig(os.path.join(save_path, title))
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError('Error:Missing parameter,four required.')
    config_file = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    if not (os.path.exists(config_file) and os.path.isfile(config_file) and os.path.exists(
            train_file) and os.path.isfile(train_file) and os.path.exists(test_file) and os.path.isfile(test_file)):
        raise ValueError('Error:File input error.')
    output_dir1 = sys.argv[4]
    dataset=  sys.argv[5]
    seed = sys.argv[6]
    num = sys.argv[7]
    num=int(num)
    seed = int(seed)
    # 设置随机数种子
    setup_seed(0)
    output_dir1 = os.path.abspath(output_dir1)
    output_dir1 = os.path.abspath(output_dir1)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    shutil.copy(config_file, output_dir1)
    start_train = time.time()
    model = get_model_and_time(train_file, output_dir1,dataset,num,seed)
    time_for_train = time.time() - start_train
    print(time_for_train)
    start_pred = time.time()
    test_model(model, test_file,output_dir1,dataset,seed)
    time_to_pred = time.time() - start_pred
    report = calculate_error(config_file, test_file, output_dir1)
    write_report(report, time_for_train, time_to_pred, output_dir1)
    calculate_gummel_test(model, config_file, output_dir1,dataset,seed)