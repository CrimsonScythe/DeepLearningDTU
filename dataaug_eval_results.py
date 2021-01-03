import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import figure, text, scatter, show
import matplotlib
def makegraphs():
       font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

       matplotlib.rc('font', **font)
       train_lst=[]
       eval_lst=[]
       trainx_lst=[]
       evalx_lst=[]

       train_lst2=[]
       train_lst

       evalaug=np.array([5.5753583908081055, 3.214068651199341, 3.7196900844573975, 3.757931709289551, 3.916210174560547, 3.392118453979492, 2.924518585205078, 4.186288356781006])
       evalnoaug=np.array([5.030195236206055, 3.3573155403137207, 3.482891798019409, 3.2040936946868896, 3.668433666229248, 3.2644119262695312, 2.2330803871154785, 2.4792068004608154])

       sns.lineplot(y=evalaug, x=np.array([1007616*1, 1007616*2, 1007616*3, 1007616*4, 1007616*5, 1007616*6,1007616*7,1007616*8]))
       sns.lineplot(y=evalnoaug, x=np.array([1007616*1, 1007616*2, 1007616*3, 1007616*4, 1007616*5, 1007616*6,1007616*7,1007616*8]))       
     
       plt.xlabel('Time steps')
       plt.ylabel('Average reward')
       plt.legend([f'random_crop      ~ {3.84}', f'no augmentation ~ {3.34}']) 
       plt.title('Test curves for 9M timesteps - Bigfish')
       plt.text(0.5, 0.5, 'matplotlib', horizontalalignment='center', verticalalignment='center')
       plt.show()
       plt.legend()

makegraphs()

