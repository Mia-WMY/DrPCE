import numpy as np
import torch
def get_deriv(X,y):
    deriv=[]
    data=torch.cat((X,y),1).detach().numpy()
    dvg, dvd, data_mat, void_mat, error_mat, is_monotone, vg, vd = monotonicitycheck(data)
    # print("data_mat", data_mat.shape)# 15 14
    # print("?",dvg.shape)
    # print("!",dvd.shape)
    derivvg, derivvd ,_,_= device_deriv(dvg, dvd, data_mat)
    for i in range(len(derivvg)):
        deriv.append([derivvg[i], derivvd[i]])
    return deriv

def get_deriv_with_level(X,y,level):
    out={}
    data=torch.cat((X,y),1).detach().numpy()
    # X[device parameter vgs vds ids]
    # vgs vds
    # print("x",X.shape)
    j=1
    dvg, dvd, data_mat, void_mat, error_mat, is_monotone, vg, vd = monotonicitycheck(data)
    # print("dvg",dvg)
    # print("dvd",dvd)

    matrix_vg=data_mat
    matrix_vd=data_mat
    while (j <= level):
        # print("data_mat", data_mat.shape)# 15 14
        deriv = []
        # derivvg, _ , matrixvg1, _ = device_deriv2(dvg, dvd, matrix_vg)
        # _, derivvd, _ ,matrixvd1  = device_deriv2(dvg, dvd, matrix_vd)
        # print("derivvg",derivvg[0:10])
        derivvg, _, matrixvg1, _ = device_deriv(dvg, dvd, matrix_vg)
        _, derivvd, _, matrixvd1 = device_deriv(dvg, dvd, matrix_vd)
        # for i in range(0,14):
        #     print(vd[i], torch.Tensor(matrixvg1)[:,i])
        for i in range(len(derivvg)):
            # print("?",derivvg[i])
            deriv.append([derivvg[i], derivvd[i]])
        out[j]=deriv
        matrix_vg= matrixvg1
        matrix_vd= matrixvd1
        # print(matrix_vd.shape)
        j=j+1

    return out
def get_deriv_with_level_gm(vg,vd,ids):
    outvg={}
    outvd= {}
    # print("vg",torch.Tensor(vg.to_numpy()))
    vg=torch.Tensor(vg.to_numpy()).reshape(-1,1)
    vd = torch.Tensor(vd.to_numpy()).reshape(-1,1)
    ids = torch.Tensor(ids.to_numpy()).reshape(-1,1)
    # print(vg.shape)
    # print(vd.shape)
    data=torch.cat((vg,vd),1)
    data = torch.cat((data, ids), 1).detach().numpy()

    # j=1
    dvg, dvd, data_mat, void_mat, error_mat, is_monotone, vg, vd = monotonicitycheck(data)
    matrix_vg=data_mat
    matrix_vd=data_mat
    derivvg, _, matrixvg1, _ = device_deriv(dvg, dvd, matrix_vg)
    _, derivvd, _, matrixvd1 = device_deriv(dvg, dvd, matrix_vd)
    # print(vg.shape)
    # print(vd.shape)
    for i in range(0,vd.shape[0]):
        outvg[vd[i]]=torch.Tensor(matrixvg1)[:,i]

    for i in range(0,vg.shape[0]):
        outvd[vg[i]]=torch.Tensor(matrixvd1)[i,:]

    return outvg,vd,outvd,vg


def monotonicitycheck(device_data):
    # print("yy", device_data.shape)
    device_data = np.array(device_data)
    # print("--", device_data)
    # device_data = device_data[:, -3:]
    vg = np.unique(device_data[:, 0])#  vg key
    # print("vg",vg)
    vd = np.unique(device_data[:, 1])# vd key
    # print("vd", len(vd))
    dvg = vg[1] - vg[0]# vg step
    dvd = vd[1] - vd[0]# vd step
    data_mat = np.ones((vg.shape[0], vd.shape[0])) * (-1000)
    # print(data_mat.shape)
    void_mat = np.zeros((vg.shape[0], vd.shape[0]))
    error_mat = np.zeros((vg.shape[0], vd.shape[0]))
    # print("??", dvg)
    # print("??", dvd)
    for row in device_data:
        # print("row",row[2]) #row[0]=vg row[1]=vd row[2]=log(ids)
        data_mat[round((row[0]-vg[0]) / dvg), round((row[1]-vd[0]) / dvd)] = row[2]
    for i in range(vg.shape[0]):
        id_temp = -1000
        for j in range(vd.shape[0]):
            if data_mat[i, j] == -1000:
                void_mat[i, j] = 1
            elif data_mat[i, j] >= id_temp:
                id_temp = data_mat[i, j]
            else:
                error_mat[i, j] = 1
                id_temp = data_mat[i, j]

    for j in range(vd.shape[0]):
        id_temp = -1000
        for i in range(vg.shape[0]):
            if data_mat[i, j] == -1000:
                void_mat[i, j] = 1
            elif data_mat[i, j] >= id_temp:
                id_temp = data_mat[i, j]
            else:
                error_mat[i, j] = 1
                id_temp = data_mat[i, j]

    if error_mat.sum() > 0:
        is_monotone = False
    else:
        is_monotone = True

    return dvg, dvd, data_mat, void_mat, error_mat, is_monotone, vg, vd


def device_deriv(dvg, dvd, data_mat):  # 15 14
    # 求导的定义是函数值的微增量关于自变量的微增量的极限
    # print("data_mat",data_mat.shape)
    # print("data_mat", data_mat)
    derivvg = (data_mat[1:, :] - data_mat[:-1, :]) / dvg  # ids增量/vg增量
    derivvd = (data_mat[:, 1:] - data_mat[:, :-1]) / dvd  # ids增量/vd增量
    # print(derivvd.shape) # 15 13
    # # 13 23
    # print(derivvg.shape) # 14 14
    # 12 24
    # print("derivvg[0,:]", derivvg[0, :].shape)  # （14，）
    # print("derivvg[1,:]", derivvg[1, :].shape)  # （14，）
    # if (derivvg.shape[0]==1 or  derivvd.shape[0]==1):
    #     continue
    derivvg_first = np.array([2 * derivvg[0, :] - derivvg[1, :]])
    derivvg_last = np.array([2 * derivvg[-1, :] - derivvg[-2, :]])
    derivvg = np.append(derivvg_first, derivvg, 0)  # 按行增加 在derivvg行前增加
    # print("derivvgx", derivvg.shape)  # （15，14）
    derivvg = np.append(derivvg, derivvg_last, 0)  # 按行增加 在derivvg行后增加
    # print("derivvgy", derivvg)  # （16，14）
    derivvg = (derivvg[1:, :] + derivvg[:-1, :]) / 2
    # print("derivvg", derivvg)  # 15，14
    matrix_grad_vg=derivvg
    derivvd_first = np.transpose(np.array([2 * derivvd[:, 0] - derivvd[:, 1]]))
    derivvd_last = np.transpose(np.array([2 * derivvd[:, -1] - derivvd[:, -2]]))
    derivvd = np.append(derivvd_first, derivvd, 1)
    derivvd = np.append(derivvd, derivvd_last, 1)
    derivvd = (derivvd[:, 1:] + derivvd[:, :-1]) / 2
    matrix_grad_vd = derivvd
    # print(derivvd.shape)  # (15,14)  # (15,14)
    derivvg = np.reshape(derivvg, -1)
    derivvd = np.reshape(derivvd, -1)
    # print("derivvg", derivvg.shape)  # 210

    return derivvg, derivvd, matrix_grad_vg, matrix_grad_vd

def device_deriv2(dvg, dvd, data_mat):  # 15 14
    # 求导的定义是函数值的微增量关于自变量的微增量的极限
    derivvg = (data_mat[2:, :] - data_mat[:-2, :]) / dvg /2 # ids增量/vg增量
    derivvd = (data_mat[:, 2:] - data_mat[:, :-2]) / dvd /2 # ids增量/vd增量
    derivvg_first = np.array([data_mat[1, :] - data_mat[0, :]]/dvg)
    derivvg_last = np.array([data_mat[-1, :] - data_mat[-2, :]]/dvg)
    derivvg = np.append(derivvg_first, derivvg, 0)  # 按行增加 在derivvg行前增加
    derivvg = np.append(derivvg, derivvg_last, 0)  # 按行增加 在derivvg行后增加
    # derivvg = (derivvg[1:, :] + derivvg[:-1, :]) / 2
    matrix_grad_vg=derivvg
    derivvd_first = np.transpose(np.array([data_mat[:, 1] - data_mat[:, 0]]/dvd))
    derivvd_last = np.transpose(np.array([data_mat[:, -1] - data_mat[:, -2]]/dvd))
    derivvd = np.append(derivvd_first, derivvd, 1)
    derivvd = np.append(derivvd, derivvd_last, 1)
    # derivvd = (derivvd[:, 1:] + derivvd[:, :-1]) / 2
    matrix_grad_vd = derivvd
    # print(derivvd.shape)  # (15,14)
    # print("derivvd", derivvd.shape)  # (15,14)
    derivvg = np.reshape(derivvg, -1)
    derivvd = np.reshape(derivvd, -1)
    # print("derivvg", derivvg.shape)  # 210

    return derivvg, derivvd, matrix_grad_vg, matrix_grad_vd

