# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 00:53:45 2021

@author: Jinzh
"""


# assume the travel time and waiting time jointly follow a bivariate Pareto distribution for now
# sample from bivariate Pareto distribution
    #ref: https://stackoverflow.com/questions/48420952/simulating-bivariate-pareto-distribution


import numpy as np
from scipy.stats import pearsonr
# class BivariatePareto():
# sample one variable, x2, from marginal distribution using inverse CDF
    
 
def pareto_inv(n, theta, a):
    '''
    

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    u = np.random.rand(n)
    x2 = theta / (u ** (1 / a))
    
    return x2

# sample the other from the conditional distribution
def pareto_cond_inv (x2, theta1, theta2, a):
    '''
    

    Parameters
    ----------
    x2 : TYPE
        DESCRIPTION.
    theta1 : TYPE
        DESCRIPTION.
    theta2 : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    u = np.random.rand(len(x2))
    x1 = theta1 + theta1 / theta2 * x2 * (1 / (u ** (1 / (a + 1))) - 1)
    
    return x1


def draw_samples(n=10000, theta1=10, theta2=15, a=5):
    '''
    

    Parameters
    ----------
    n : TYPE, optional
        DESCRIPTION. The default is 1e3.
    theta1 : TYPE, optional
        DESCRIPTION. The default is 0.15.
    theta2 : TYPE, optional
        DESCRIPTION. The default is 0.1.
    a : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    '''
    
    x2 = pareto_inv(int(n), theta2, a)
    x1 = pareto_cond_inv(x2, theta1, theta2, a)
    
    return x1, x2


def plot_density(x1, x2):
    # libraries & dataset
    import seaborn as sns
    import matplotlib.pyplot as plt
    # set seaborn style
    sns.set_style("white")

    # x1
    sns.kdeplot(x1)
    plt.xlabel('x1')
    plt.show()   
    # plt.hist(x1)
    # plt.xlabel('x1')
    # plt.show()
    
    # x2
    sns.kdeplot(x2)
    plt.xlabel('x2')
    plt.show()
    # plt.hist(x2)
    # plt.xlabel('x2')
    # plt.show()
    
    # Basic 2D density plot
    sns.kdeplot(x1, x2, cmap="Blues", shade=False, thresh=0)
    plt.xlim([2, 35])
    plt.ylim([2, 35])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()



def main():
    
    n = 1000
    theta1 = 10
    theta2 = 10
    a = 1.6
    
    np.random.seed(321123)        
    x1, x2 = draw_samples(n, theta1, theta2, a)
    
    cor_x1_x2,_ = pearsonr(x1, x2)
    print('\n===== The correlation between x1 and x2 is: {} ====='.format(cor_x1_x2))
    
    plot_density(x1, x2)


main()