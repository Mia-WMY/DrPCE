import numpy as np
import os
from tools.kop_calculator import KOP_calculator
pred_r_suffix = 'pr.txt'
id_min=1e-20
error_test_file = ['364', '449']
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


def write_report(result,  output_dir):
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write('\n'.join(result))

dataset=['circle','rectangle','triangle']
for set in dataset:
    seed=1
    for poly in range(1,15):
        config_file = fr'/Users/chococolate/Desktop/new/{set}/config.txt'
        test_file = fr'/Users/chococolate/Desktop/new/{set}/parametertest.txt'
        output_dir=f'{set}400_seed_{seed}_{poly}'
        if os.path.exists(output_dir):
            print(output_dir)
            report = calculate_error(config_file, test_file, output_dir)
            write_report(report, output_dir)