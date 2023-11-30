import csv
import numpy as np
csv_read = csv.reader(open('/Users/chococolate/Desktop/dataset/res/res_csv/results_10_2_0.5_18_11.csv'))
x=0
index=0.05
y=np.arange(0, 0.7, 0.05)
k=0
for i in csv_read:
    y=0
    for j in i:
        str="{:<8}	{:<8}	{:<8}".format(round(x,2),round(y,2),j)
        print(str)
        # print(round(x,2),"	",round(y,2),"	",j)
        y=y+index
    x=x+index