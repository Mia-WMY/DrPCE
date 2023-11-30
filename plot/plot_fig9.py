import math
import numpy as np
import matplotlib.pyplot as plt
# GAA-rectangle
from matplotlib import ticker
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.rcParams["font.family"] = "Times New Roman"
dim=8
leg_param=[]
x1=[1,2,3,4,5,6,7,8,9,10]
for i in x1:
	leg_param.append(int(math.factorial(i+dim) / math.factorial(i) / math.factorial(dim)))

leg=[0.0743605,0.043229485,0.018647,0.014840045,0.007859194,0.007352857,0.0051437406,0.003555969,0.003252097,0.002392798]

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1, 2))  # 设置指数的显示范围

fig, (ax1,ax2) = plt.subplots(1, 2)
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
ax1.set_xlim(1,10)
ax2.set_xlim(1,10)
ax1.set_xticks(range(1, 11, 2))
ax1.set_xticklabels(range(1, 11, 2))
ax2.set_xticks(range(1, 11, 2))
ax2.set_xticklabels(range(1, 11, 2))

ax1.plot(x1,leg,'bo-',label='PCE')
l1=ax1.axhline(y=0.00205102, color='green',linestyle=':',label='DrPCE-Net(q=2)')
l2=ax1.axhline(y=0.00129391, color='red',linestyle=':',label='DrPCE-Net(q=3)')

ax1.legend(loc=(0.18, 0.72),fontsize=14)
ax1.text(2, 0.00220,'0.0021',fontdict={'fontsize':14},color='green')
ax1.text(2, 0.00140,'0.0013',fontdict={'fontsize':14},color='red')
ax2.plot(x1,leg_param,'bo-',label='PCE',linewidth=2.0)

l2=ax2.axhline(y=1357, color='green', linewidth=2.0,linestyle=':',label='DrPCE-Net(q=2)')
l2=ax2.axhline(y=3121, color='red', linewidth=2.0,linestyle=':',label='DrPCE-Net(q=3)')
ax2.text(7.5, -500,'1357',fontdict={'fontsize':14},color='green')
ax2.text(7.5, 3700,'3121',fontdict={'fontsize':14},color='red')
ax1.text(1.2,0.070,'0.074',fontdict={'fontsize':14},color='blue')
ax1.text(7.8,0.0022,'0.0024',fontdict={'fontsize':14},color='blue')

ax1.set_title('(a)',fontsize=16)
ax2.set_title('(b)',fontsize=16)
ax2.yaxis.set_major_formatter(formatter)
ax1.tick_params(axis='both', direction='in', labelsize=16)
ax2.tick_params(axis='both', direction='in', labelsize=16)
plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.95, wspace=0.2, hspace=0.27)

plt.xlim((1, 10))
plt.xlabel('Degree of expansion, q',fontsize=18)
ax1.set_ylabel('RMSE',fontsize=18)

ax1.set_xlabel('Degree of expansion, q',fontsize=18)
ax2.set_ylabel('Number of parameters',fontsize=18)
ax1.set_yscale('log')

ax2.legend(bbox_to_anchor=(0.87, 0.92),fontsize=14)
plt.savefig('param.pdf')
plt.show()
