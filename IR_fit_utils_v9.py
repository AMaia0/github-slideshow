#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


# In[2]:


def one_oscillator(w, delta_eps, w0, gamma):
    """
    Complex permittivity at frequency w of one oscillator.
    
    Arguments: 
    w -- array with frequencies in cm-1 
    delta_eps --
    w0 --
    gamma --
    
    Returns:
    eps -- array with complex permittivity at freqs w
    """
    
    eps = (delta_eps*(w0**2))/(w0**2 - w**2 - gamma*w*1j)
    
    return eps


# In[3]:


def permittivity(w, *args):
    """
    Computes the complex permittivity at frequency w as a sum of oscillators. The number of 
    oscillators is implied in parameters. 
    
    Arguments: 
    w -- frequency in cm-1
    parameters -- dictionary containing eps_inf and delta_eps, w0 and gamma for each oscillator

    
    Returns:
    eps -- complex permittivity at freq w. Will be used to compute reflectivity. 
    
    """
    
    # Extract parameters from args 
    
    parameters_array = np.array(args) # convert from tuple to array
    n = (len(parameters_array)-1)//3  # number of oscillators
    
    eps_inf = np.abs(parameters_array[0])
    W0 = np.abs(parameters_array[(n+1):(2*n+1)])   
    Delta_eps = np.abs(parameters_array[1:(n+1)]/W0) # parameters_array[1:(n+1)] = delta_eps*w0 
    Gamma = np.abs(parameters_array[(2*n+1):])
    
    #parameters_dict = {'eps_inf': eps_inf, 'delta_eps': delta_eps, 'w0': w0, 'gamma': gamma}
    
    eps = eps_inf
    
    for i in range(0, n):
        eps += one_oscillator(w, Delta_eps[i], W0[i], Gamma[i])
    
    
        
    return eps


# In[4]:


def reflectivity(w, *args):
    """
    Computes reflectivity from complex permitivitty.
    
    Arguments: 
    w -- frequency in cm-1
    eps_inf -- real number 
    parameters -- (3, number of oscillators)
    each column of parameters should have delta_eps, w0 and gamma 
    
    Returns:
    R -- reflectivity at freq w. Will be used to compute cost. 
    """
    
    eps = permittivity(w, *args)
    
    temp1 = np.absolute(np.sqrt(eps) - 1)**2
    temp2 = np.absolute(np.sqrt(eps) + 1)**2
    
    R = temp1/temp2
    
    
    return R


# In[5]:


def get_exp_data(file_name, freq_range = (None, None) , double_check = False):
        
    """
    Gets experimental data from specified file. First column is frequency and second one is reflectivity.
    
    Arguments: 
    file_name -- string with the name of the file to be imported
    freq_range -- tuple (min_freq, max_freq), if not set to None, it allows to select just a portion of the exp spectra
    double_check -- if set to True, prints data-frame and first 6 elements to make sure everything is ok.

    Returns:
    w_exp -- array with frequencies
    R_exp -- array with reflectivity 
                     
    """
    # accomodate the cases where the columnds are separate by tab and space 
    exp_data_df = pd.read_csv(file_name, sep = ' ', header = None)
    
    if len(exp_data_df.columns) < 2:
        exp_data_df = pd.read_csv(file_name, sep = '\t', header = None)
        #assert(exp_data_df.columns < 2), 'Check spacing between columns'   

    w_exp = np.array(exp_data_df[0])
    R_exp = np.array(exp_data_df[1])

    # consider frequencies and reflectivity corresponding to lower_bound < w_exp < upper_bound  
    min_freq, max_freq = freq_range
    if min_freq == None and max_freq == None:
        pass
    else:
        w_exp_temp = w_exp[w_exp > min_freq]
        R_exp_temp = R_exp[np.nonzero(w_exp > min_freq)]
        w_exp = w_exp_temp[w_exp_temp < max_freq]
        R_exp = R_exp_temp[np.nonzero(w_exp_temp < max_freq)]
        assert(w_exp.shape == R_exp.shape) # make sure both arrays have the same shape
    
    # print dataframe to double check
    if double_check == True:  
        print(exp_data_df)
        print('first 6 elements of w_exp = ' + str(w_exp[:6]))
        print('first 6 elements of R_exp = ' + str(R_exp[:6]))
        
    
    return w_exp, R_exp


# In[6]:


def get_initial_guess(file_name, from_wasf = True, double_check = False):
    
    """
    Gets initial values of eps_inf, delta_eps, w0 and gamma from a text file. 
    
    Such text file can be the output from wasf (in the vertical arrangement!) or have the following format:
        1st column is eps_inf, 2nd column is delta_eps, 3rd column is w0, 4th column is gamma
    
    Arguments: 
    file_name -- string with the name of the file to be imported 
    from_wasf -- True/False 
    double_check -- if set to True, prints intermediate steps to make sure everything is ok

    Returns:
    initial_guess -- array of the form [eps_inf, delta_eps_1, ..., delta_eps_n, w0_1, ..., w0_n, gamma_1, ..., gamma_n]
                     to be passed to the optimization routine, where n is the number of oscillators
                     
    """
    if from_wasf:
        
        # create data-frame with eps1, eps2, mu1, m2 (separately because of the way wasf creates the file)
        eps_df = pd.read_csv(file_name, sep = '\t', skiprows = 12, nrows = 4, header = None)
    
        eps_inf = np.array(eps_df[1])[0]
    
        # create data-frame with oscillator parameters
        initial_guess_df = pd.read_csv(file_name, sep = '\t', skiprows = 16, header = None) 
        initial_guess_df = initial_guess_df.drop(0, axis = 1) #delete one useless column
    
        delta_eps = np.array(initial_guess_df[2])
        w0 = np.array(initial_guess_df[4])
        gamma = np.array(initial_guess_df[6])
        
    else: 
        
        initial_guess_df = pd.read_csv(file_name, sep = '\t')
        eps_inf = np.array(initial_guess_df['eps_inf'])[0]
        delta_eps = np.array(initial_guess_df['delta_eps'])
        w0 = np.array(initial_guess_df['w0'])
        gamma = np.array(initial_guess_df['gamma'])
        

    initial_guess = np.append(eps_inf, [delta_eps*w0, w0, gamma])
    
    # visualize data-frame to make sure everything is ok
    if double_check: 
        print(initial_guess_df) 
        print('eps_inf = ' + str(eps_inf))
        print('delta_eps = ' + str(delta_eps))
        print('w0 = ' + str(w0))
        print('gamma = ' + str(gamma))
        print('there are ' + str(len(w0)) + ' oscillators')
    
    return initial_guess


# In[7]:

def save_reflectivity_and_permittivity(w, R, eps, save_dir, sample_name, temperature):
    '''
    Saves reflectivity and permittivity (real and imaginary parts) to a text file located in save_dir.
    
    Arguments:
        w -- frequencies at which to calculate the complex permittivity
        R -- array with reflectivity data
        eps -- array with permittivity data
        save_dir -- directory where folder with output file will be created
        sample_name -- name of the sample 
        temperature -- current temperature (will be written in the name of the file)
    
    '''
     
    # Create dataframe to store the data
    header = ['Frequency (cm-1)', 'Reflectivity', 'Re[eps]', 'Im[eps]']
    eps_and_R_df = pd.DataFrame(columns = header)
    
    # Store data in columns
    eps_and_R_df['Frequency (cm-1)'] = w
    eps_and_R_df['Reflectivity'] = R
    eps_and_R_df['Re[eps]'] = eps.real
    eps_and_R_df['Im[eps]'] = eps.imag
    
    # Create folder in save_dir where the file will be saved, if it doesn't exist already
    path_eps_and_R = os.path.abspath(os.path.join(save_dir, sample_name + '_R_and_eps'))
    if not os.path.exists(path_eps_and_R):
        os.makedirs(path_eps_and_R)
    
    # Name of the file
    file_name = os.path.join(path_eps_and_R, sample_name + '_R_and_eps' + '_' + str(temperature) + 'K.txt')
    
    # save to file
    eps_and_R_df.to_csv(file_name, sep = '\t', index = False)

# In[8]:

def get_data_to_fit(path_data_to_fit, ascending_order_T):
    
    
    '''
    Gets all the data to fit from files in path_data_to_fit and retrieves the corresponding temperature from each file's name.
    The names of those files should have the following format:'whatever_000K.extension'
    It's important that the temperature has always three digits and is followed by 'K.extension' (123K.dat, 004K.txt, ...)
    
    file name's written differently WILL NOT be imported!
    
    If ascending_order_T is set to True (False), data will be fitted in ascending (descending) temperature order. 
    '''
    
    
    # We need to through the exp data to import and assign to each file it's temperature
    # Also, let's be sure that there are no other files. To do that, remove any file without 'K.'
    files_data_to_fit = [s for s in os.listdir(path_data_to_fit) if 'K.' in s] 
    
    # Create a dictionary with the names of the files (keys) and corresponding temperature
    files_and_temperatures_dict = {}
    for file_name in files_data_to_fit:
        index = file_name.find('K.')
        temperature = int(file_name[index-3:index])
        files_and_temperatures_dict[str(temperature)] = file_name
    
    # Create array with temperatures in descending or ascending order
    temperatures_array = np.array(list(files_and_temperatures_dict.items()))[:,0] # remove keys
    temperatures_array = [int(T) for T in temperatures_array] # convert from str to int
    temperatures_array.sort(reverse = not(ascending_order_T))


    return temperatures_array, files_and_temperatures_dict

        
 # In[9]: 

def organize_fit_parameters(*args):
    
    '''
    Takes as input an array of the form:
        [eps_inf, delta_eps_1*w0_1, ..., delta_eps_n*w0_n, w0_1, ..., w0_n, gamma_1, ..., gamma_n]
        
    and returns an array of the form:
        [eps_inf, delta_eps_1, w0_1, gamma_1, ..., delta_eps_n, w0_n, gamma_n]
    
    '''
    parameters = np.array(args) 
    
    # Number of oscillators
    assert (len(parameters) - 1)%3 == 0, 'number of oscillators is not ok' 
    n = (len(parameters) - 1) // 3
    
    # Extract all parameters and separate them into arrays
    eps_inf = parameters[0]
    W0 = parameters[(n+1):(2*n+1)]
    Delta_eps = parameters[1:(n+1)]/W0   # parameters_array[1:(n+1)] = delta_eps*w0
    Gamma = parameters[(2*n+1):]
    
    # Create a matrix where rows are Delta_eps, W0 and Gamma. Then transpose it and flatten it. The resulting array has the desired order. 
    new_array = np.array([Delta_eps, W0, Gamma]).T.flatten()
    
    # Append eps_inf 
    new_array = np.append(eps_inf, new_array)  
    
    return new_array

# In[10]:
    
def save_fit_parameters_single_temperature(save_dir, sample_name, temperature, *args):
    
    '''
    Creates dataframe to store the fit parameters for a given temperature and then saves it as text file in a folder created in save_dir.
    1st column is eps_inf, 2nd column is delta_eps, 3rd column is w0, 4th column is gamma.
    
    
    '''
    
    # Convert to array  [eps_inf, delta_eps_1, ..., delta_eps_n, w0_1, ..., w0_n, gamma_1, ..., gamma_n]
    parameters = np.array(args) 
    
    # Create dataframes
    header_fits = ['eps_inf', 'delta_eps', 'w0', 'gamma']
    parameters_df = pd.DataFrame(columns = header_fits)
   
    # Number of oscillators
    assert (len(parameters) - 1)%3 == 0, 'number of oscillators is not ok' 
    n = (len(parameters) - 1) // 3
    
    # Fill each column of dataframe
    parameters_df['eps_inf'] = parameters[0]*np.ones(n)           
    parameters_df['w0'] = parameters[(n+1):(2*n+1)]            
    parameters_df['delta_eps'] = parameters[1:(n+1)]/parameters_df['w0']   # parameters_array[1:(n+1)] = delta_eps*w0
    parameters_df['gamma'] = parameters[(2*n+1):]  

    # Create folder in work_dir where plots will be saved (if it doesn't exist already)
    path_fit_params = os.path.abspath(os.path.join(save_dir, sample_name + '_fit_params'))
    if not os.path.exists(path_fit_params):
        os.makedirs(path_fit_params)
        
    # Create name of the file where dataframe will be saved
    file_name = sample_name + '_fit_params_' + str(temperature) + 'K.txt'
    
    # Save dataframe to specified path as text file with file_name 
    parameters_df.to_csv(os.path.join(path_fit_params, file_name), sep = '\t', index = False)
    
    # Let's also save the fit parameters in a format that can be loaded into wasf
    parameters_wasf = import_params_to_wasf(os.path.join(path_fit_params, file_name))
    # Save new file as .wsf
    file_name_wasf = sample_name + '_fit_params_wasf_' + str(temperature) + 'K.wsf'
    with open(os.path.join(path_fit_params, file_name_wasf), 'w') as new_file:
        new_file.writelines(parameters_wasf)

# In[11]:   
    
def plot_and_save_fig(x, y, labels, work_dir, sample_name, temperature, save = True):
    '''
    
    Arguments:
        x -- array with x-values
        y -- tuple (y_scatter, y_line), where (x, y_scatter) will be a scatter plot and (x, y_line) will be a line plot
        labels -- (label_scatter, label_line, guess_or_fit), where label_scatter and label_line are, respectively, the 
                  legend labels for the scatter and line plots and guess_or_fit = 'guess', 'fit' are labels that specify
                  if the line plot will correspond to an initial guess or to a fit
        work_dir -- str, working directory where to create a folder to save plots
        sample_name -- str, name of the sample
        temperature -- str, corresponding temperature (will appear in file name for the plot)
        save -- bool, saves figure in a folder created in work_dir if set to True
    
    
    '''
    # 'initial_guess @ ' + str(temperature) + ' K'
    # 'exp_data @ ' + str(temperature) + ' K'
    
    y_scatter, y_line = y
    label_scatter, label_line, guess_or_fit = labels
    
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    
    # Axis Labels
    plt.ylabel('Reflectivity')
    plt.xlabel('Frequency (cm-1)')
    
    # Scatter Plot
    plt.scatter(x, y_scatter, label = label_scatter, color = 'blue')
    
    # Line Plot             
    plt.plot(x, y_line, label = label_line, color = 'red')
    
    # Legend size and location
    plt.legend(loc = "upper right", prop={'size': 16}, bbox_to_anchor=(1.3, 0.6))
    
    # Set background to be white
    fig.patch.set_facecolor('xkcd:white')
    
    plt.show()
    
    if save:
        # Create folder in work_dir where plots will be saved (if it doesn't exist already)
        path_plots = os.path.abspath(os.path.join(work_dir, sample_name + '_plots'))
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)
        # Name of figure
        fig_name = sample_name + '_' + guess_or_fit + '_' + str(temperature) + ' K'    
        # Save figure to folder in path_plots
        fig.savefig(os.path.join(path_plots, fig_name), bbox_inches='tight')
    
# In[12]:
    
def import_params_to_wasf(file_name):
    
    '''
    comment!
    
    '''
    
    # Get data from file 
    parameters_df = pd.read_csv(file_name, sep = '\t')
    eps_inf = np.array(parameters_df['eps_inf'])[0]
    delta_eps = np.array(parameters_df['delta_eps'])
    w0 = np.array(parameters_df['w0'])
    gamma = np.array(parameters_df['gamma'])
    
    # number of oscillators
    n = len(delta_eps)
    assert len(delta_eps) == len(w0), 'check number of oscillators'

    # Create a text file with the proper format for wasf to import 
    
    header = ['{*************************************************}\n',
              '{***  copyright university of stuttgart, 2006  ***}\n',
              '{***  written by Steffen Schultz               ***}\n',
              '{***  email: steffen.schultz@gmx.de            ***}\n',
              '{*************************************************}\n',
              '\n',
              '{*************************************************}\n',
              '{*** DO NOT EDIT THIS FILE!                    ***}\n',
              '{*************************************************}\n',
              '\n',
              '  <models>\n',
              '    numberOfModels = 1\n',
              '    <model[1]>\n',
              '      name = Model_1\n']

    epsilon =  ['      <epsilon>\n',
                '        eps1 = ' + str(eps_inf) + '\n',
                '        lock_eps1 = true\n',
                '        eps2 = 0\n',
                '        lock_eps2 = true\n',
                '        mu1 = 1\n',
                '        lock_magn1 = true\n',
                '        mu2 = 0\n',
                '        lock_magn2 = true\n',
                '      </epsilon>\n']

    new_contents = header + epsilon

    for term_num in range(n):
        # repeat this part term_num times, one for each oscillator 
        term = ['      <term[' + str(term_num + 1) + ']>\n',
                '        type = 1\n',
                '        value[0] = ' + str(delta_eps[term_num]) + '\n',
                '        lock[0] = false\n',
                '        value[1] = ' + str(w0[term_num]) + '\n',
                '        lock[1] = false\n',
                '        value[2] = ' + str(gamma[term_num]) + '\n',
                '        lock[2] = false\n',
                '      </term[' +str(term_num + 1) + ']>\n',]
        new_contents += term


    footer = ['    </model[1]>\n',
     '  </models>\n',
     '\n',
     '  <profiles[1]>\n',
     '    profileLayerContainerNr = 1\n',
     '    <layers>\n',
     '      numberOfLayers = 1\n',
     '      <layer[1]>\n',
     '        name = Layer_1\n',
     '        modelNr = 1\n',
     '        thickness = 1\n',
     '      </layer[1]>\n',
     '    </layers>\n',
     '    <options>\n',
     '      optionNr = 0\n',
     '    </options>\n',
     '  </profiles[1]>\n',
     '  <profiles[2]>\n',
     '    profileLayerContainerNr = 1\n',
     '    <layers>\n',
     '      numberOfLayers = 1\n',
     '      <layer[1]>\n',
     '        name = Layer_1\n',
     '        modelNr = 1\n',
     '        thickness = 1\n',
     '      </layer[1]>\n',
     '    </layers>\n',
     '    <options>\n',
     '      optionNr = 0\n',
     '    </options>\n',
     '  </profiles[2]>\n',
     '  <profiles[3]>\n',
     '    profileLayerContainerNr = 1\n',
     '    <layers>\n',
     '      numberOfLayers = 1\n',
     '      <layer[1]>\n',
     '        name = Layer_1\n',
     '        modelNr = 1\n',
     '        thickness = 1\n',
     '      </layer[1]>\n',
     '    </layers>\n',
     '    <options>\n',
     '      optionNr = 0\n',
     '    </options>\n',
     '  </profiles[3]>\n',
     '  <profiles[4]>\n',
     '    profileLayerContainerNr = 1\n',
     '    <layers>\n',
     '      numberOfLayers = 1\n',
     '      <layer[1]>\n',
     '        name = Layer_1\n',
     '        modelNr = 1\n',
     '        thickness = 1\n',
     '      </layer[1]>\n',
     '    </layers>\n',
     '    <options>\n',
     '      optionNr = 0\n',
     '    </options>\n',
     '  </profiles[4]>\n',
     '  <profiles[5]>\n',
     '    profileLayerContainerNr = 1\n',
     '    <layers>\n',
     '      numberOfLayers = 1\n',
     '      <layer[1]>\n',
     '        name = Layer_1\n',
     '        modelNr = 1\n',
     '        thickness = 1\n',
     '      </layer[1]>\n',
     '    </layers>\n',
     '    <options>\n',
     '      optionNr = 0\n',
     '    </options>\n',
     '  </profiles[5]>\n',
     '  <profiles[6]>\n',
     '    profileLayerContainerNr = 1\n',
     '    <layers>\n',
     '      numberOfLayers = 1\n',
     '      <layer[1]>\n',
     '        name = Layer_1\n',
     '        modelNr = 1\n',
     '        thickness = 1\n',
     '      </layer[1]>\n',
     '    </layers>\n',
     '    <options>\n',
     '      optionNr = 0\n',
     '    </options>\n',
     '  </profiles[6]>\n',
     '  <profiles[7]>\n',
     '    profileLayerContainerNr = 1\n',
     '    <layers>\n',
     '      numberOfLayers = 1\n',
     '      <layer[1]>\n',
     '        name = Layer_1\n',
     '        modelNr = 1\n',
     '        thickness = 1\n',
     '      </layer[1]>\n',
     '    </layers>\n',
     '    <options>\n',
     '      optionNr = 0\n',
     '    </options>\n',
     '  </profiles[7]>\n',
     '  <profiles[8]>\n',
     '    profileLayerContainerNr = 1\n',
     '    <layers>\n',
     '      numberOfLayers = 1\n',
     '      <layer[1]>\n',
     '        name = Layer_1\n',
     '        modelNr = 1\n',
     '        thickness = 1\n',
     '      </layer[1]>\n',
     '    </layers>\n',
     '    <options>\n',
     '      optionNr = 0\n',
     '    </options>\n',
     '  </profiles[8]>\n',
     '\n',
     '\n',
     '\n']
    
    new_contents += footer
    
    
    return new_contents 

    
# In[13]:
    
def fit_reflectivity(sample_name, save_dir, path_data_to_fit, initial_guess, freq_range, ascending_order_T, method, bounds, max_iter):
    
    '''
    Starting at the temperature for which an initial guess has been provided, the code below
    will find the optimal parameters for that temperature. Those parameters will then be
    used as an initial guess for the next temperature.
    
    Arguments:
        sample_name -- Name of the sample (will appear in all the names of the files).
        save_dir -- Directory of the folder in which to save all output files.
        path_data_to_fit -- Directory of folder with reflectivity data to be fitted.
 
        files_and_temperatures_dict -- Dictionary with file names and corresponding temperatures. 
        temperatures_array -- Array with temperatures in files_and_temperatures_dict.
        
        initial_guess -- initial guess for (eps_inf, delta_eps_1, ..., delta_eps_n, w0_1, ..., w0_n, gamma_1, ..., gamma_n) 
        
        freq_range -- tuple (min_freq, max_freq), if not set to None, it allows to select just a portion of the exp spectra 
        method -- Optimization algorithm to use. Can be 'lm' or 'trf', where only 'trf' handles bounded parameters.
        bounds -- Tupple with lower and upper bounds on the parameters (must be consistent with the chosen method).
        max_iter -- Maximum number of iterations of method.
    
    Outputs:
        plots of ... --
        eps data -- 
        text file with all the fit parameters for every temperature to be saved at work_dir
        
    '''
    
    # Get current working directory
    work_dir = os.getcwd()
    
    # Get data to fit from files in path_data_to_fit and the corresponding temperatures (taken from each file's name)
    temperatures_array, files_and_temperatures_dict = get_data_to_fit(path_data_to_fit, ascending_order_T)
    
    # Set working directory to be the folder with data to fit (revert back to previous one after loop)
    os.chdir(path_data_to_fit)

    i = 0
    choice = 'y'
    while choice == 'y' and i < len(temperatures_array):
    
        # set current temperature
        temperature = temperatures_array[i]
        # get exp data from file 
        file_name_exp = files_and_temperatures_dict[str(temperatures_array[i])]
        w_exp, R_exp = get_exp_data(file_name_exp, freq_range, double_check = False)
    
        # Calculate reflectivity from the parameters specified in initial guess 
        R_initial_guess = reflectivity(w_exp, *initial_guess)
    
    
        # Plot experimental data and initial guess
        labels = ('exp_data @ ' + str(temperature) + ' K', 'initial_guess @ ' + str(temperature) + ' K', 'guess')
        plot_and_save_fig(w_exp, (R_exp, R_initial_guess), labels, save_dir, sample_name, temperature, save = True)
    
        # Obtain optimal parameters
        fit_parameters, cov = curve_fit(reflectivity, xdata = w_exp, ydata = R_exp, 
                                        bounds = bounds, p0 = initial_guess, 
                                        method = method, maxfev = max_iter)
        
        # Use previous result as the next initial guess (keep negative parameters, it's ok)
        initial_guess = fit_parameters
        
        # Calculate reflectivity from fit parameters
        R_fit = reflectivity(w_exp, *fit_parameters)
        eps_fit = permittivity(w_exp, *fit_parameters)
        
        # Plot experimental data and fit
        labels = ('exp_data @ ' + str(temperature) + ' K', 'fit @ ' + str(temperature) + ' K', 'fit')
        plot_and_save_fig(w_exp, (R_exp, R_fit), labels, save_dir, sample_name, temperature, save = True)
        plt.show()
        
        # R and eps were calculated using np.abs(fit_parameters), so let's save that instead of fit_parameters:
        fit_parameters = np.abs(fit_parameters)
        # Save these parameters to a text file (for each temperature)
        save_fit_parameters_single_temperature(save_dir, sample_name, temperature, *fit_parameters)
        
        # Save reflectivity and permittivity data calculated from fit_parameters to a text file in save_dir
        save_reflectivity_and_permittivity(w_exp, R_fit, eps_fit, save_dir, sample_name, temperature)
        
        # activate if you want to check the procedure at each temperature
        #choice = input('Continue? (y/n)')
    
        i += 1
    
    # Go back to original working directory
    os.chdir(work_dir)
    
    


    
    
