# 递归获取.csv文件存入到list1
import os
import numpy as np
import csv
def txt_csv(index,csv_file,shape):
    xf = open(fr'{index}.txt', 'w')
    csv_read = csv.reader(open(csv_file))
    x = 0
    index = 0.01
    y = np.arange(0, 0.8, 0.01)
    k = 0
    str = "{:<8}	{:<8}	{:<8}".format('vg', 'vd', 'ids')
    # print(str)
    xf.write(str)
    xf.write('\n')
    for i in csv_read:
        y = 0
        for j in i:
            str = "{:<8}	{:<8}	{:<8}".format(round(x, 2), round(y, 2), j)
            # print(str)
            xf.write(str)
            xf.write('\n')
            # print(round(x,2),"	",round(y,2),"	",j)
            y = y + index
        x = x + index
# 将所有文件的路径放入到listcsv列表中
def list_dir(file_dir,shape):
    # list_csv = []
    f = open(fr'index.txt', 'w')
    ftrain = open(fr'parametertrain.txt', 'w')
    ftest = open(fr'parametertest.txt', 'w')
    dir_list = os.listdir(file_dir)
    index=0
    if shape=='rectangle':
        str = "{},{},{},{},{},{},{}".format('lg_nw', 'w_nw','h_nw', 't_ox', 'N_sd', 'eps_ox','index')
    else:
        str = "{},{},{},{},{},{}".format('lg_nw', 'r_nw', 't_ox', 'N_sd', 'eps_ox', 'index')
    f.write(str)
    f.write('\n')
    ftrain.write(str)
    ftrain.write('\n')
    ftest.write(str)
    ftest.write('\n')

    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if os.path.splitext(path)[1] == '.csv':
            csv_file = os.path.join(file_dir, cur_file)
            csv_path=os.path.join(file_dir, cur_file)
            csv_file=csv_file.split('.csv')[0]
            csv_file=csv_file.split('results_')[1]
            at1=csv_file.split('_')[0]
            at2=csv_file.split('_')[1]
            at3= csv_file.split('_')[2]
            if shape=='rectangle':
                at4 = csv_file.split('_')[3]
                e1 = 10 ** float(csv_file.split('_')[4])
                e1 = "%.1e" % e1
                at5 = csv_file.split('_')[5]
                str = "{},{},{},{},{},{},{}"\
                    .format(at1, at2, at3,at4, e1, at5, index)
                print(str)
                if index < 300:
                    ftrain.write(str)
                    ftrain.write('\n')
                else:
                    ftest.write(str)
                    ftest.write('\n')
                f.write(str)
                f.write('\n')
                txt_csv(index, csv_path,shape)
            else:
                e1=10**float(csv_file.split('_')[3])
                e1="%.1e" % e1
                at5=csv_file.split('_')[4]
                str = "{},{},{},{},{},{}".format(at1,at2,at3,e1,at5,index)
                print(str)
                if index<300:
                    ftrain.write(str)
                    ftrain.write('\n')
                else:
                    ftest.write(str)
                    ftest.write('\n')
                f.write(str)
                f.write('\n')
                txt_csv(index, csv_path,shape)
            # print(str)
            # f.write(str)
            index=index+1
    return list_csv


if __name__ == '__main__':
    shape = 'circle'
    paths = fr'/Users/chococolate/Desktop/res_csv'
    list_csv = []
    # os.mkdir(f'{shape}')
    list_dir(file_dir=paths,shape=shape)
    # print(list_csv)