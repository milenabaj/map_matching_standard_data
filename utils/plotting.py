"""
@author: Milena Bajic (DTU Compute)
"""
import sys, os, pickle, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplleaflet
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator,LogFormatter)

    
def plot_geolocation(longitudes = None, latitudes = None, name = 'plot', out_dir = '.', plot_firstlast=0, plot_html_map = True, title=None, full_filename = None, preload = False):
            
    # Name
    if full_filename:
        name = full_filename.replace('.pickle','.png')
    else:
        name = '{0}/{1}_map.png'.format(out_dir, name)
    
    
    # Matplotlib figure
    if os.path.exists(name) and preload:
        pickle_name = name.replace('.png','.pickle')
        print('Loading ',pickle_name)
        with open(pickle_name,'rb') as f:
            fig = pickle.load(f)
        #plt.show()
        
    else:
        print('Plotting')
        
        # Figure, axis
        fig,ax = plt.subplots()
        
        # Title
        if title:
            ax.set_title(title)
           
        # Plot first/last
        if plot_firstlast!=0:
            ax.scatter(longitudes[0:plot_firstlast], latitudes[0:plot_firstlast], s = 50, c='red',marker='o',alpha=0.3, label = 'Start')
            ax.scatter(longitudes[0:2], latitudes[0:2], s = 90, c='red',marker='x',alpha=0.5, label = 'Start')
            
            ax.scatter(longitudes[-plot_firstlast:], latitudes[-plot_firstlast:], s = 50, c='black',marker='o',alpha=0.3, label = 'End') 
            ax.scatter(longitudes[-2:], latitudes[-2:], s = 90, c='black',marker='x',alpha=0.5, label = 'End') 
         
        # Plot else
        ax.scatter(longitudes, latitudes, s = 8, c='dodgerblue',marker='o',alpha=0.3)
        ax.legend()
        
        # Save as png
        fig.savefig(name)
        print('Figure saved: {0}'.format(name)) 
              
        # Save also as pickle
        fig_name = name.replace('.png','.pickle')
        with open(fig_name,'wb') as f:
            pickle.dump(fig, f)  
        
    # Html map
    if plot_html_map:
        from selenium import webdriver
        
        print('Will try to open web browser')
        html_name = name.replace('.png','.html')
        
        # Html can't plot with legend
        if fig.axes[0].get_legend():
            fig.axes[0].get_legend().remove()
            
        # If webbrowser available, plot it
        mplleaflet.display(fig, tiles='cartodb_positron')
        mplleaflet.show(fig, html_name)
        print('File saved: {0}'.format(html_name))
         
        # Save pdf prinout of the webpage
        html_link = 'file://{0}/{1}'.format(os.getcwd(),html_name)
        printout_name = name.replace('.png','_printout.png')
        #print(html_link)
        
        browser = webdriver.Firefox()
        browser.get(html_link)
        
        #Give the map tiles some time to load
        time.sleep(10)
        browser.save_screenshot(printout_name)
        browser.quit()

    return fig


def plot_geolocation_2D(gps_points_1, gps_points_2, name = None, out_dir='.', plot_firstlast=1):
    (lon1,lat1) = gps_points_1
    (lon2,lat2) = gps_points_2
    
    fig,ax=plt.subplots()
    
    ax.scatter(lon1, lat1, s = 15, c='dodgerblue',marker='o',alpha=0.6)
    if plot_firstlast!=0:
        ax.scatter(lon1[0:plot_firstlast], lat1[0:plot_firstlast], s = 50, c='yellow',marker='o',alpha=1)
    
    ax.scatter(lon2, lat2, s = 8, c='yellow',marker='o',alpha=0.3) 
    if plot_firstlast!=0:
        ax.scatter(lon2[-plot_firstlast:], lat2[-plot_firstlast:], s = 50, c='black',marker='o',alpha=0.4) 
     
    # Name
    if name:
        mplleaflet.show(path='{0}/map_{1}.html'.format(out_dir, name))
    else:
        mplleaflet.show() 
    return
 
    
