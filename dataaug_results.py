import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
def makegraphs(filename1, filename2, name):
       train_lst=[]
       trainx_lst=[]
       train_lst2=[]

       font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

       matplotlib.rc('font', **font)

       file=open(filename2, 'r')
       data = file.read()
       for s in data.split('\n'):
              if ('Mean' in s):
                train_lst.append(float(re.split(r'([0-9]+\.[0-9]+)', s)[-2]))
                trainx_lst.append(int(re.split(r'([0-9]+)', s)[1]))
           
       
       file=open(filename1, 'r')
       data = file.read()
       for s in data.split('\n'):
            train_lst2.append(float(re.split(r'([0-9]+\.[0-9]+)', s)[-2]))


       sns.lineplot(y=np.array(train_lst2), x=np.array(trainx_lst))
       sns.lineplot(y=np.array(train_lst), x=np.array(trainx_lst))
    
   
       plt.xlim(right=9650176)
       plt.xlabel('Time steps')
       plt.ylabel('Average reward over 3 epochs')
       plt.legend([f'augmentation ~ {np.mean(train_lst2)}',
                   f'no augmentation      ~ {np.mean(train_lst)}']) 
       plt.title(f'Training curves for 9M timesteps - {name}')
       
       plt.show()
       plt.legend()
       

       print(np.mean(train_lst))
       print(np.mean(train_lst2))
       

makegraphs(filename1='bigfish_cutout_color.txt', filename2='bigfish_no_augmentation.txt', name='BigFish')
makegraphs(filename1='starpilotaug.txt', filename2='starpilot_noaug.txt', name='StarPilot')
