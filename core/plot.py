# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:29:15 2022

@author: Thom Badings
"""

import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt

def threshold_histogram(thresholds, N, outfile, bins=25):
    
    print('-- Create histogram of thresholds obtained using Theorem 1 (N='+
          str(N)+'...')
    
    sns.histplot(thresholds, binwidth=0.25, binrange=[13,18])
    
    plt.xlabel('Threshold on spec.')
    plt.ylabel('Count')
    
    plt.xlim(13, 18)
    plt.xticks([13,14,15,16,17,18])
    plt.yticks([0,200,400,600])
    
    plt.tight_layout()
    plt.show()
        
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    
    print('-- Histogram exported to: '+str(outfile))