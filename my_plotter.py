import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

def my_bar( data, category ):
    
    bar_data = round( 100*data[category].value_counts()/len(data), 1 )
    labels = list( np.sort( bar_data.index))
    heights = bar_data[labels]
    
    plt.bar( labels, heights, alpha = 0.7, width = 0.6 )
    plt.grid(True, alpha = 0.0)
    plt.ylabel("Percentage", fontsize=14)
    plt.title("Distribution of {text}".format(text=category))
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim([0,max(heights)*1.2])
    
    for (i,j) in zip(labels,heights):
        plt.annotate( str(j), xy = (i,j),
                     xytext = (i,j+0.1*max(heights)), ha = "center",
                     bbox = {'boxstyle': 'round', 'pad':0.5, 'facecolor':'orange', 
                             'edgecolor':'orange', 'alpha':0.6},
                     arrowprops={'arrowstyle':'wedge,tail_width=0.5',
                                'alpha':0.6, 'color':'orange'})
        

'''
def my_hist( data, category, number_of_bins=10):
    
    hist_data = data[category]
    
    plt.hist( hist_data, bins = number_of_bins)
'''
