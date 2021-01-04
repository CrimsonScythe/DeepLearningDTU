import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
def makegraphs(filename1, filename2, name='Title'):
       train_lst=[]
       trainx_lst=[]
       train_lst2=[]
       trainx_lst2=[]

       font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

       matplotlib.rc('font', **font)
       
       fig, ax = plt.subplots()

       file=open(filename2, 'r')
       data = file.read()
       for i,s in enumerate(data.split('\n')):
            train_lst.append(float(s))
            trainx_lst.append(int(8192*(i+1)))
           
       
       file=open(filename1, 'r')
       data = file.read()
       for i,s in enumerate(data.split('\n')):
            train_lst2.append(float(s))
            trainx_lst2.append(int(i+1))


       ax.set_ylabel('Normalized reward')
       ax.set_xlabel('Timesteps')
     #   ax.legend('Training curve')
     #   ax2.legend('Evaluation curve')
       ax=sns.lineplot(y=np.array(train_lst), x=trainx_lst, label='Training curve')

       ax2=ax.twiny()
       ax2.get_xaxis().set_visible(False)


       ax2=sns.lineplot(data=np.array(train_lst2), color='orange', label='Evaluation curve')
     #   ax2.legend()

    #    plt.xlim(right=9650176)
    #    plt.xlim(right=10000000)
    #    plt.xlabel('Time steps')
    #    plt.ylabel('Average reward over 3 epochs')
       plt.legend(bbox_to_anchor=(0.29,0.9))
     #   plt.legend([f'augmentation ~ {np.mean(train_lst2)}',
               #     f'no augmentation      ~ {np.mean(train_lst)}']) 
       plt.title('Training and evaluation curves')
       
       plt.show()
       plt.legend()
       

    #    print(np.mean(train_lst))
    #    print(np.mean(train_lst2))
       

makegraphs(filename1='dataeval1.txt', filename2='data1.txt')
# makegraphs(filename1='starpilotaug.txt', filename2='starpilot_noaug.txt', name='StarPilot')
