'''
初始化 KOP_calculator 实例，分别输入config文件和data文件，
调用实例方法get_kop()得到 vtgm,idsat,idmax,gmmax,ioff的元组（ioff=-1时表示不存在）
'''
import numpy as np
import scipy.interpolate as spi
import re


def diff_forward(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    return dy / dx


def find_x_for_y(x, y, yval):
    delta_y = y - yval
    tmp_ind = np.argmin(np.abs(delta_y))
    if delta_y[tmp_ind] < 0:
        ind = tmp_ind
        if ind == y.shape[0] - 1:
            ind = y.shape[0] - 2
    else:
        ind = tmp_ind - 1
        if ind < 0:
            ind = 0
    if y[ind + 1] - y[ind] > 1e-3 * yval:
        x_diff_val = (x[ind + 1] - x[ind]) / (y[ind + 1] - y[ind])
        x_val = x[ind] + (yval - y[ind]) * x_diff_val
    else:
        x_val = (y[ind + 1] + y[ind]) * 0.5
    return x_val


def Idvg_kop(vg, id, vds):
    gm = np.abs(diff_forward(vg, id))
    idx = np.argmax(gm)
    vt_gm = vg[idx] - id[idx] / gm[idx] - 0.5 * vds
    idmax = np.max(id)
    return gm, vt_gm, idmax,


def get_config_info(file):
    with open(file) as f:
        lines = f.readlines()
    vdd = ''
    vdlin = ''
    step = 0.1
    for line in lines:
        line = line.strip()
        infos = re.split('\s+', line)
        try:
            for i, info in enumerate(infos):
                if 'vdd' == info and len(infos) > i + 1:
                    vdd = float(infos[i + 1])
                    if infos[i + 1] == '0.5':
                        step = 0.05
                if 'vdlin' == info and len(infos) > i + 1:
                    vdlin = float(infos[i + 1])
        except Exception as e:
            raise ValueError('Error:%s,probable not vdd or vdlin in file %s' % (str(e), file))
    if (not isinstance(vdd, float)) or (not isinstance(vdlin, float)):
        raise ValueError('Error:not vdd or vdlin in file %s' % file)
    return vdd, vdlin, step


def get_id_vg_ioff(file, vdlin, vdd):
    with open(file) as f:
        first_line = f.readline()
    if re.match('vg', first_line):
        data = np.loadtxt(file, dtype='float', delimiter='\t', skiprows=1)
    else:
        data = np.loadtxt(file, dtype='float', delimiter=',', skiprows=3)
    valid_data = data[data[:, 1] == vdlin]
    vgs = valid_data[:, 0]
    ids = valid_data[:, 2]
    ioff = -1
    ioff_data = data[np.logical_and(data[:, 0] == 0, data[:, 1] == vdd)]
    if ioff_data.shape[0] != 0:
        ioff = ioff_data[0, 2]
    return ids, vgs, ioff


class KOP_calculator:
    def __init__(self, config_file, data_file, ifloor=1e-14, ntype=1, interp='log'):
        self.ifloor = ifloor
        self.ntype = ntype
        self.interp = interp
        self.set_data(config_file, data_file)

    def set_data(self, config_file, data_file):
        self.vdd, self.vdlin, self.step = get_config_info(config_file)
        id, vg, self.ioff = get_id_vg_ioff(data_file, self.vdlin, self.vdd)
        self.vg1 = self.vdd
        self.vg0 = 0
        if self.ntype < 0.5:
            vg = -vg
        if vg[5] < vg[0]:
            vg = np.flip(vg)
            id = np.flip(id)
        id = np.abs(id)
        if self.interp == 'log':
            self.interf = spi.interp1d(vg, np.log10(id), kind='cubic', bounds_error=False, fill_value='extrapolate')
        else:
            self.interf = spi.interp1d(vg, id, kind='cubic', bounds_error=False, fill_value='extrapolate')
        num = int((self.vg1 - self.vg0) / self.step)
        self.vg = self.step * np.linspace(0, num, num + 1) + self.vg0
        self.id = self.interf(self.vg)
        self.idsat = self.interf(self.vdd).reshape(-1)[0]
        if self.interp == 'log':
            self.id = np.power(10, self.id)
            self.idsat = np.power(10, self.idsat)
        self.valididx = self.id > self.ifloor
        self.gm, self.vtgm, self.idmax = Idvg_kop(self.vg, self.id, self.vdlin)
        self.gmmax = np.max(self.gm)

    def get_kop(self):
        return self.vtgm, self.idsat, self.idmax, self.gmmax, self.ioff


if __name__ == '__main__':
    kop = KOP_calculator(r'/Users/chococolate/Dropbox/TCAD/data/GAA_data/config.txt', r'/Users/chococolate/Dropbox/TCAD/data/GAA_data/1.txt')
    print(kop.get_kop())
