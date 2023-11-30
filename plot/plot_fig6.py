import numpy as np
import torch
import os
import pandas as pd
from matplotlib import pyplot as plt, ticker
import matplotlib.animation as animation
from sklearn.metrics import r2_score
from tools.physics import get_deriv_with_level_gm
plt.rcParams["font.family"] = "Times New Roman"
def custom_format(value, pos):
        return "%.1f" % value
def four_plot_log4_Ids(list1,list2,list3):
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((-1, 1))  # 设置指数的显示范围
    index1, dir_path1, test_path1=list1[0],list1[1],list1[2]
    index2, dir_path2, test_path2 = list2[0], list2[1], list2[2]
    index3, dir_path3, test_path3 = list3[0], list3[1], list3[2]
    true_pth = dir_path1 + f'/{index1}.txt'
    pred1_pth = test_path1 + f'/{index1}pr.txt'
    true_pth2 = dir_path2 + f'/{index2}.txt'
    pred1_pth2 = test_path2 + f'/{index2}pr.txt'
    true_pth3 = dir_path3 + f'/{index3}.txt'
    pred1_pth3 = test_path3 + f'/{index3}pr.txt'
    fig, axes = plt.subplots(3,4,figsize=(12,6))
    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=12)
    ax1=axes[0,0]
    ax2=axes[0,1]
    ax3=axes[0, 2]
    ax4= axes[0, 3]
    ax5 = axes[1, 0]
    ax6 = axes[1, 1]
    ax7 = axes[1, 2]
    ax8 = axes[1, 3]
    ax9 = axes[2, 0]
    ax10 = axes[2, 1]
    ax11 = axes[2, 2]
    ax12 = axes[2, 3]
    ax1.set_xlim(0,0.7)
    ax2.set_xlim(0,0.7)
    ax3.set_xlim(0, 0.7)
    ax4.set_xlim(0,0.7)
    ax5.set_xlim(0, 0.7)
    ax6.set_xlim(0, 0.7)
    ax7.set_xlim(0, 0.7)
    ax8.set_xlim(0, 0.7)
    ax9.set_xlim(0, 0.7)
    ax10.set_xlim(0, 0.7)
    ax11.set_xlim(0, 0.7)
    ax12.set_xlim(0, 0.7)
    ####
    true_df = pd.read_table(true_pth, sep='\t')
    pred1_df= pd.read_table(pred1_pth, skiprows=3, sep=',')
    pred1_df.columns = ['vgs', 'vds', 'ids']
    true_df.columns = ['vg', 'vd', 'ids']
    #
    true_df2 = pd.read_table(true_pth2, sep='\t')
    pred1_df2 = pd.read_table(pred1_pth2, skiprows=3, sep=',')
    pred1_df2.columns = ['vgs', 'vds', 'ids']
    true_df2.columns = ['vg', 'vd', 'ids']
    #
    true_df3 = pd.read_table(true_pth3, sep='\t')
    pred1_df3 = pd.read_table(pred1_pth3, skiprows=3, sep=',')
    pred1_df3.columns = ['vgs', 'vds', 'ids']
    true_df3.columns = ['vg', 'vd', 'ids']
    #
    pred1_df.loc[pred1_df['vds'] == 0, 'ids'] = 0
    pred1_df2.loc[pred1_df['vds'] == 0, 'ids'] = 0
    pred1_df3.loc[pred1_df['vds'] == 0, 'ids'] = 0
    ####
    outvg,vd,outvd,vg = get_deriv_with_level_gm(pred1_df['vgs'],pred1_df['vds'],pred1_df['ids'])
    toutvg, tvd, toutvd,tvg= get_deriv_with_level_gm(true_df['vg'], true_df['vd'], true_df['ids'])
    vd_set = {0.05, 0.7}
    vg_set = {0.4,0.5,0.6,0.7}
    for vd_val in vd_set:
        true_subset = true_df[true_df['vd'] == vd_val]
        pred1_subset = pred1_df[pred1_df['vds'] == vd_val]
        ax1.plot(pred1_subset['vgs'],
                 pred1_subset['ids']*1e6,
                 color='blue',
                 linewidth=2,)
        ax1.scatter(true_subset['vg'],
                 true_subset['ids']*1e6,
                 color='blue', marker="o",facecolors='none',  s=35)
        ax1.set_yscale('log')
    for i in vd:  # gm
        if (np.allclose(i, 0.05, rtol=1e-3) or np.allclose(i, 0.7, rtol=1e-3)):
            ax2.plot(vg, outvg[i] * 1e6, color='blue',linewidth=2)
            ax2.scatter(vg, toutvg[i] * 1e6, color='blue', marker="o", facecolors='none', s=35)
    for vg_val in vg_set:
        true_subset = true_df[true_df['vg'] == vg_val]
        pred1_subset = pred1_df[pred1_df['vgs'] == vg_val]
        ax3.plot(pred1_subset['vds'],
                 pred1_subset['ids']*1e6,
                 color='blue',
                 linewidth=2,)
        ax3.scatter(true_subset['vd'],
                    true_subset['ids']*1e6,
                    color='blue', marker="o",facecolors='none',  s=35)
        ax3.set_ylim(-2*1e-2, 20*1e-1)
        ax3.set_yticks(np.linspace(0,3.5,6))

    for i in vg: # gds
        if (np.allclose(i, 0.3, rtol=1e-3) or np.allclose(i, 0.5, rtol=1e-3) or np.allclose(i, 0.4, rtol=1e-3) or np.allclose(i, 0.6, rtol=1e-3)):
            ax4.plot(vd, outvd[i]*1e6, color='blue',linewidth=2)
            ax4.scatter(vd, toutvd[i]*1e6, color='blue',marker="o",facecolors='none',s=35)
            ax4.set_ylim(-2 * 1e-2, 8)
            ax4.set_yticks(np.linspace(-1, 20, 4))
    outvg2, vd2, outvd2, vg2 = get_deriv_with_level_gm(pred1_df2['vgs'], pred1_df2['vds'], pred1_df2['ids'])
    toutvg2, tvd2, toutvd2, tvg2 = get_deriv_with_level_gm(true_df2['vg'], true_df2['vd'], true_df2['ids'])
    vd_set2 = {0.05, 0.7}
    vg_set2 = {0.4, 0.5, 0.6, 0.7}
    for i in vd2:  # gm
        if (np.allclose(i, 0.05, rtol=1e-3) or np.allclose(i, 0.7, rtol=1e-3) ):
            ax6.plot(vg2, outvg2[i]*1e6, color='red',linewidth=2)
            ax6.scatter(vg2, toutvg2[i]*1e6,color='red', marker="o",facecolors='none',  s=35)
    for i in vg2:  # gds
        if (np.allclose(i, 0.3, rtol=1e-3) or np.allclose(i, 0.5, rtol=1e-3) or np.allclose(i, 0.4, rtol=1e-3) or np.allclose(i, 0.6, rtol=1e-3)):
            ax8.plot(vd2, outvd2[i]*1e6, color='red',linewidth=2)
            ax8.scatter(vd2, toutvd2[i]*1e6,  color='red', marker="o",facecolors='none',  s=35)
            ax8.set_ylim(-2 * 1e-2, 22)
            ax8.set_yticks(np.linspace(-1, 26, 4))
    for vd_val in vd_set2:
        true_subset2 = true_df2[true_df2['vd'] == vd_val]
        pred1_subset2 = pred1_df2[pred1_df2['vds'] == vd_val]
        ax5.plot(pred1_subset2['vgs'],
                 pred1_subset2['ids']*1e6,
                 color='red',linewidth=2)
        ax5.scatter(true_subset2['vg'],
                    true_subset2['ids']*1e6,
                    color='red', marker="o", facecolors='none', s=35)
        ax5.set_yscale('log')
    for vg_val in vg_set2:
        true_subset2 = true_df2[true_df2['vg'] == vg_val]
        pred1_subset2 = pred1_df2[pred1_df2['vgs'] == vg_val]
        ax7.plot(pred1_subset2['vds'],
                 pred1_subset2['ids']*1e6,
                 color='red',linewidth=2)
        ax7.scatter(true_subset2['vd'],
                    true_subset2['ids']*1e6,
                    color='red', marker="o",  facecolors='none',s=35)
        ax7.set_ylim(-2 * 1e-2, 6)
        ax7.set_yticks(np.linspace(0,6, 4))
    ##
    outvg3, vd3, outvd3, vg3 = get_deriv_with_level_gm(pred1_df3['vgs'], pred1_df3['vds'], pred1_df3['ids'])
    toutvg3, tvd3, toutvd3, tvg3 = get_deriv_with_level_gm(true_df3['vg'], true_df3['vd'], true_df3['ids'])
    vd_set3 = {0.05,0.7}
    vg_set3 = {0.4, 0.5, 0.6, 0.7}  # rectangle
    for i in vd3:  # gm
        if (np.allclose(i, 0.05, rtol=1e-3) or np.allclose(i, 0.7, rtol=1e-3) ):
            ax10.plot(vg3, outvg3[i]*1e6, color='green',linewidth=2)
            ax10.scatter(vg3, toutvg3[i]*1e6,  color='green', marker="o", facecolors='none', s=35)
    for i in vg3:  # gds
        if (np.allclose(i, 0.4, rtol=1e-3) or np.allclose(i, 0.5, rtol=1e-3) or np.allclose(i, 0.6, rtol=1e-3) or np.allclose(i, 0.7, rtol=1e-3)):
            ax12.plot(vd3, outvd3[i]*1e6, color='green',linewidth=2 )
            ax12.scatter(vd3, toutvd3[i]*1e6,   color='green', marker="o", facecolors='none', s=35)
            ax12.set_ylim(-2 * 1e-2, 125)
            ax12.set_yticks(np.linspace(-7, 125, 4))
    for vg_val in vg_set3:
        true_subset3 = true_df3[true_df3['vg'] == vg_val]
        pred1_subset3 = pred1_df3[pred1_df3['vgs'] == vg_val]
        ax11.plot(pred1_subset3['vds'],
                  pred1_subset3['ids']*1e6,
                  color='green',linewidth=2)
        ax11.scatter(true_subset3['vd'],
                     true_subset3['ids']*1e6,
                     color='green', marker="o", s=35, facecolors='none')
        ax11.set_ylim(2, 26)
        ax11.set_yticks(np.linspace(0, 25, 6))
    for vd_val in vd_set3:
        true_subset3 = true_df3[true_df3['vd'] == vd_val]
        pred1_subset3 = pred1_df3[pred1_df3['vds'] == vd_val]
        ax9.plot(pred1_subset3['vgs'],
                 pred1_subset3['ids']*1e6,
                 color='green',linewidth=2)
        ax9.scatter(true_subset3['vg'],
                    true_subset3['ids']*1e6,
                    color='green', marker="o", s=35, facecolors='none')
        ax9.set_yscale('log')
    ax1.text(0.4, 1e-8, f"Triangle", size=16, ha='left', va='center',color='black')
    ax5.text(0.34, 1e-6, f"Rectangle", size=16, ha='left', va='center',color='black')
    ax9.text(0.4, 1e-2, f"Circle", size=16, ha='left', va='center',color='black')
    # ########
    ax1.text(0.13,1e-1, r"$V_{ds}$=0.7V", size=14, ha='left', va='center')
    ax1.text(0.4,1e-4, r"$V_{ds}$=0.05V", size=14, ha='left', va='center')
    ##
    ax2.text(0.23, 10, r"$V_{ds}$=0.7V", size=14, ha='left', va='center')
    ax2.text(0.05, 2.2, r"$V_{ds}$=0.05V", size=14, ha='left', va='center')
    # ##
    ax3.text(0.2, 1.7, r"$V_{gs}$=0.4V to 0.7V", size=14, ha='left', va='center')
    ax3.text(0.2, 0.7, f"with 0.1-V step", size=14, ha='left', va='center')
    # ##
    ax4.text(0.15, 8, r"$V_{gs}$=0.4V to 0.7V", size=14, ha='left', va='center')
    ax4.text(0.15, 3.5, f"with 0.1-V step", size=14, ha='left', va='center')
    # ########
    ax5.text(0.13, 2, r"$V_{ds}$=0.7V", size=14, ha='left', va='center')
    ax5.text(0.4, 1e-2, r"$V_{ds}$=0.05V", size=14, ha='left', va='center')
    # ##
    ax6.text(0.13, 15, r"$V_{ds}$=0.7V", size=14, ha='left', va='center')
    ax6.text(0.03, 4, r"$V_{ds}$=0.05V", size=14, ha='left', va='center')
    ##
    ax7.text(0.23, 4.1, r"$V_{gs}$=0.4V to 0.7V", size=14, ha='left', va='center')
    ax7.text(0.23, 2.8, f"with 0.1-V step", size=14, ha='left', va='center')
    # ##
    ax8.text(0.14, 15, r"$V_{gs}$=0.4V to 0.7V", size=14, ha='left', va='center')
    ax8.text(0.2, 11, f"with 0.1-V step", size=14, ha='left', va='center')
    ##
    ax9.text(0.1, 10, r"$V_{ds}$=0.7V", size=14, ha='left', va='center')
    ax9.text(0.4, 1, r"$V_{ds}$=0.05V", size=14, ha='left', va='center')
    ##
    ax10.text(0.3, 30, r"$V_{ds}$=0.7V", size=14, ha='left', va='center')
    ax10.text(0.34, 18, r"$V_{ds}$=0.05V", size=14, ha='left', va='center')
    ##
    ax11.text(0.23, 10.6, r"$V_{gs}$=0.4V to 0.7V", size=14, ha='left', va='center')
    ax11.text(0.23, 5, f"with 0.1-V step", size=14, ha='left', va='center')
    ##
    ax12.text(0.22, 80, r"$V_{gs}$=0.4V to 0.7V", size=14, ha='left', va='center')
    ax12.text(0.22, 60, f"with 0.1-V step", size=14, ha='left', va='center')
    # ax1.set_xlabel(f'Vgs(uV)',fontdict={'fontsize': 16})
    ax1.set_ylabel(r'$I_{ds}$ (uA)',fontdict={'fontsize': 16})
    # # ax2.set_xlabel('Vgs(V)',fontdict={'fontsize': 16})
    ax2.set_ylabel('$g_m$ (uA/V)',fontdict={'fontsize': 16})
    # # ax3.set_xlabel('Vds(V)',fontdict={'fontsize': 16})
    ax3.set_ylabel(r'$I_{ds}$ (uA)',fontdict={'fontsize': 16})
    # # ax4.set_xlabel('Vds(V)',fontdict={'fontsize': 16})
    ax4.set_ylabel(r'$g_{ds}$ (uA/V)',fontdict={'fontsize': 16})
    # # ax5.set_xlabel(f'Vgs(uV)',fontdict={'fontsize': 16})
    ax5.set_ylabel(r'$I_{ds}$ (uA)',fontdict={'fontsize': 16})
    # # ax6.set_xlabel('Vgs(V)',fontdict={'fontsize': 16})
    ax6.set_ylabel(r'$g_m$ (uA/V)',fontdict={'fontsize': 16})
    # # ax7.set_xlabel('Vds(V)',fontdict={'fontsize': 16})
    ax7.set_ylabel(r'$I_{ds}$ (uA)',fontdict={'fontsize': 16})
    # # ax8.set_xlabel('Vds(V)',fontdict={'fontsize': 16})
    ax8.set_ylabel('$g_{ds}$ (uA/V)',fontdict={'fontsize': 16})
    ax9.set_xlabel(r'$V_{gs}$ (V)',fontdict={'fontsize': 16})
    ax9.set_ylabel(r'$I_{ds}$ (uA)',fontdict={'fontsize': 16})
    ax10.set_xlabel(r'$V_{gs}$ (V)',fontdict={'fontsize': 16})
    ax10.set_ylabel(r'$g_m$ (uA/V)',fontdict={'fontsize': 16})
    ax11.set_xlabel(r'$V_{ds}$ (V)',fontdict={'fontsize': 16})
    ax11.set_ylabel(r'$I_{ds}$ (uA)',fontdict={'fontsize': 16})
    ax12.set_xlabel(r'$V_{ds}$ (V)',fontdict={'fontsize': 16})
    ax12.set_ylabel('$g_{ds}$ (uA/V)',fontdict={'fontsize': 16})
    plt.subplots_adjust(left=0.06,right=0.98,bottom=0.1,top=0.95,wspace=0.35,hspace=0.27)
    save_fig_pth ='result.pdf'
    plt.gcf().savefig(save_fig_pth)
    plt.show()

dir_path1 = fr'../data/circle'
dir_path2 = fr'../data/rectangle'
dir_path3 = fr'../data/triangle'
test1_path=fr'../results/DrPCE'#circe=4
test2_path=fr'../results/DrPCE'
test3_path=fr'../results/DrPCE'

special_i=[]
list1=[402]
list1.append(dir_path1)
list1.append(test1_path)
list2=[]
list2.append(482)
list2.append(dir_path2)
list2.append(test2_path)
list3=[]
list3.append(309)
list3.append(dir_path3)
list3.append(test3_path)
four_plot_log4_Ids(list3,list2,list1)
