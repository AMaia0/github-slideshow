# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:14:30 2021

@author: André
"""

# In[1]:

import os
import numpy as np
import tkinter
from tkinter import filedialog

# Select folder where IR_fit_utils_vX is located
root = tkinter.Tk()
root.withdraw() #use to hide tkinter window
ir_module_path = filedialog.askdirectory(parent = root, title = 'Import IR_fit_utils file')

# Import IR_fit_utils_v?
os.chdir(ir_module_path)
import IR_fit_utils_v9 as ir 

# In[2]:

# Choose initial guess
# Specify if text file from which to import the initial guess is an output from wasf (True) or this code (False)
from_wasf = True

# Select text file to import
root = tkinter.Tk()
root.withdraw() #use to hide tkinter window
file_name_guess = filedialog.askopenfilename(parent = root, title = 'Select initial guess')
    
# Get first initial guess
initial_guess = ir.get_initial_guess(file_name_guess, from_wasf, double_check = False)

# In[3]:

# Choose folder with exp data to import
root = tkinter.Tk()
root.withdraw() #use to hide tkinter window
path_data_to_fit = filedialog.askdirectory(parent = root, title = 'Select folder with data to fit')

# In[4]:

# Choose folder in which to save all output files
root = tkinter.Tk()
root.withdraw() #use to hide tkinter window
save_dir = filedialog.askdirectory(parent = root, title = 'Select folder where files will be saved')    
    
# In[5]:

# Specify the sample's name (will appear in all the file's names)
sample_name = '24Abr_BiMnCuO_3_lm'

# Select if fitting should be done in ascending or descending order of temperatures
ascending_order_T = True

# Choose frequency range (fit only a portion of the exp spectra)
min_freq = 5
max_freq = 651
freq_range = (min_freq, max_freq)

# Choose between 'lm', 'trf' or 'dogbox' methods
method = 'lm'
bounds = (-np.inf, np.inf) # do not change

# Choose bounds for parameters (lm only works for unconstrained problems!)
if method == 'trf':
    # set all parameters to be positive and allow eps_inf to only vary by +/- x%
    eps_inf = initial_guess[0]
    #lower_bounds = np.append(0.97*eps_inf, -0.0001*np.ones(len(initial_guess)-1)) # feel free to change
    #upper_bounds = np.append(1.15*eps_inf, np.inf*np.ones(len(initial_guess)-1)) # feel free to change
    lower_bounds = np.append(6, -0.0001*np.ones(len(initial_guess)-1)) # feel free to change
    upper_bounds = np.append(7, np.inf*np.ones(len(initial_guess)-1)) # feel free to change
    bounds = (lower_bounds, upper_bounds)

    
    
# Choose maximum number of  iterations
max_iter = 1000000


# In[6]:

# Start the fitting procedure
ir.fit_reflectivity(sample_name, save_dir, path_data_to_fit, initial_guess, freq_range, ascending_order_T, method, bounds, max_iter)
print('\n' + 'Done!')





 # In[ ]:


'''

Starting at the temperature for which an initial guess has been provided, the code below
will find the optimal parameters for that temperature. Those parameters will then be
used as an initial guess for the next temperature. It's possible to choose if the fitting
should start from the highest or lowest temperature available by setting ascending_order 
to True or False. 

A text file will be created in the working directory with the optimal parameters. 

In the working directory, two folders will be created. One with the plots of the initial 
guess and fit for each temperature and another with the permittivity for each temperature.

The name of the files with exp data MUST end in the following format:

    temperatureK.extension, where temperature must have 3 digits

For example, sample_004K.txt or sample_300K.txt


'''


