import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
#plt.rcParams['figure.dpi'] = 1600
#plot data using the evluated the data sheets
file_name = 55
data = np.array([np.loadtxt("20x20 FIG\\20x20.%d.csv"%file_name,delimiter=",",skiprows=1)[:,0:11],
        np.loadtxt("30x30 FIG\\30x30.%d.csv"%file_name,delimiter=",",skiprows=1)[:,0:11],
        np.loadtxt("50x50 FIG\\50x50.%d.csv"%file_name,delimiter=",",skiprows=1)])
plt1 = np.mean(data[0][:,1:]*100,axis=1)
plt2 = np.mean(data[1][:,1:]*100,axis=1)
plt3 = np.mean(data[2][:,1:]*100,axis=1)

plt.plot(data[0][:,0],plt1,label="$Grid_{20x20} \, (55\, \%)$")
plt.plot(data[1][:,0],plt2,label="$Grid_{30x30} \, (55\, \%)$")
plt.plot(data[2][:,0],plt3,label="$Grid_{50x50} \, (55\, \%)$")
max_val = np.around(np.max(np.array([plt1,plt2,plt3])),2)
print(max_val)

plt.yticks(np.arange(0,80,5))
plt.ylim(0,80)
plt.xlabel("$iteration \; number  \; (unit \; time)$")
plt.ylabel("$Red \;weights \;(\%)$")
plt.legend(loc='lower right')
plt.savefig("%d-Fig"%file_name)


'''
#initlize weight file 
grid =  [r".\30x30\55",r".\30x30\60",r".\30x30\55",r".\30x30\55"]
filenames = ["30x30.55.fallout_edges","30x30.60.fallout_edges","30x30.55.fallout_edges","30x30.55.fallout_edges"]
labels = ["55% rate","60% rate","55% rate","55% rate"]
for idx,g in enumerate(grid):
    count = 0
    cascade_percentage = []
    for root, dirs, files in os.walk(g, topdown=False):
        for name in files:
            fn = os.path.join(root, name)
            name_blocks = fn.split("_")
            if "falloff_edges" in fn and int(name_blocks[3])%2==0:
                count += 1
                cascade_edges_red = {}
                cascade_edges_black = {}
                cascade_edges_total= {}
                data = np.genfromtxt(fn,delimiter=",",encoding='utf-8')
                data = np.concatenate([data[1:,0].reshape(-1,1),data[1:,-1].reshape(-1,1)],axis=1)
                for row in data:
                    iter_cout=int(row[0])
                    if iter_cout not in cascade_edges_total.keys():
                        if row[1] == 1:
                            cascade_edges_black[iter_cout] = 1
                            cascade_edges_red[iter_cout] = 0
                            cascade_edges_total[iter_cout] = 1
                        else:
                            cascade_edges_black[iter_cout] = 0
                            cascade_edges_red[iter_cout] = 1
                            cascade_edges_total[iter_cout] = 1
                    else:
                        if row[1] == 1:
                            cascade_edges_black[iter_cout] = cascade_edges_black[iter_cout] + 1
                            cascade_edges_total[iter_cout] = cascade_edges_total[iter_cout] + 1
                        else:
                            cascade_edges_red[iter_cout] = cascade_edges_red[iter_cout] + 1
                            cascade_edges_total[iter_cout] = cascade_edges_total[iter_cout] + 1

                df_black = pd.DataFrame.from_dict(cascade_edges_black, orient="index")
                df_red = pd.DataFrame.from_dict(cascade_edges_red, orient="index")
                df_total = pd.DataFrame.from_dict(cascade_edges_total, orient="index")
                df = pd.concat([df_black,df_red,df_total],axis=1)
                df.to_csv(g+"\\"+filenames[idx]+str(count)+".csv")
'''

