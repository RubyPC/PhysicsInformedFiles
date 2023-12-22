#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:35:00 2023

@author: ruby
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
# Opening dark matter dataset
import os
import glob

wdir = '/Users/ruby/Documents/Python Scripts/PhysicsInformedFeatures/'

# open the dataset
data_path = wdir+'Data/'

no_sub = np.load(data_path+'nosubstructure.npy')
cdm = np.load(data_path+'colddarkmatter.npy')

# functions for generating a pixel grid given the lensed image and for plotting
# lensed images

profile_size = 64

def coords(profile_size, min_angle_vision=-3.232, max_angle_vision=3.232):
    # check if the profile_size is a positive integer > 1
    if not isinstance(profile_size, int) or profile_size <= 1:
        raise ValueError("profile_size must be a positive integer greater than 1.")
    
    # check if min_angle_vision < max_angle_vision
    if min_angle_vision >= max_angle_vision:
        raise ValueError("min_angle_vision must be less than max_angle_vision")
    
    # generate coordinate grids for image positions
    x = np.linspace(min_angle_vision, max_angle_vision, profile_size)
    y = np.linspace(min_angle_vision, max_angle_vision, profile_size)
    
    # calculate the pixel width of the image
    pixel_width = x[1] - x[0]
    
    return x, y, pixel_width

    
# estimate the sersic, bn, constant given the sersic index, n.
def sersic_bn_constant(n):
    # calculate the bn constant
    bn = 1.999 * n - 0.327
    
    return bn    
    
# generate the intensity distribution of light in an elliptical galaxy using 
# the sersic profile    
def sersic_profile(x, y, x0=0., y0=0., theta=0., q=0.7522, n=1., r_ser=0.3, I0=1.):
 
    # calculate the radii from the center of the ellipse
    R = np.sqrt(((np.cos(theta)*(x-x0) + np.sin(theta)*(y-y0))**2
                + (q**2)*((np.sin(theta)*(x-x0) - np.cos(theta)*(y-y0))**2))/q)
    
    # calculate the constant bn based on the Sersic index
    bn = sersic_bn_constant(n)
    
    # calculate the intensity at each point (x,y) based on the Sersic profile
    Is = I0 * np.exp(-bn*((R/r_ser)**(1/n)))
    return Is    

# function to compute the difference between the observed sersic profile and
# our estimation    
def residuals(params, x, y, sersic_observed):
    # get the parameters of the sersic profile
    x0, y0, theta, q, n, r_ser = params
    sersic_model = sersic_profile(x, y, x0, y0, theta, q, n, r_ser, I0=1.)
    
    return (sersic_model - sersic_observed).ravel()    

# function to fit a sersic profile to the observed data   
def fit_sersic_profile(x, y, sersic_observed, initial_guess, min_angle_vision=-3.232,
                       max_angle_vision=3.232):
  
    # bounds for parameters where q is constrained to lie in [0,1]
    bounds = ([min_angle_vision, min_angle_vision, 0, 0, 0.5, 0.1],
              [max_angle_vision, max_angle_vision, np.pi, 1, 10.0, 3.0])
    result = least_squares(residuals, initial_guess, args=(x, y, sersic_observed), bounds=bounds)
    
    return result.x     

# estimate the sersic parameters for a given profile
def estimate_parameters(x, y, source_profile, profile_size):
   
    # create a meshgrid of x and y coordinates
    Y, X = np.meshgrid(x, y)
    
    # fitting the Sersic profile to the meshgrid data
    x0, y0, theta, q, n, r_ser = fit_sersic_profile(X.ravel(), Y.ravel(), source_profile.ravel(),
                                                         initial_guess=[0, 0, 0, 1, 0.5, 1])
    
    # returning the estimated parameters
    return x0, y0, theta, q, n, r_ser    
    
# create the coordinate grid for plotting
x, y, pixel_width = coords(profile_size=profile_size)    
    
# estimate the sersic parameters for cold dark matter   
x0_c, y0_c, theta_c, q_c, n_c, r_ser_c = estimate_parameters(x, y, cdm, profile_size=profile_size)    
# and for the lens with no dark matter substructure
x0_n, y0_n, theta_n, q_n, n_n, r_ser_n = estimate_parameters(x, y, no_sub, profile_size=profile_size)

print(f'The Sersic parameters for CDM profile: x0={x0_c}, y0={y0_c}, theta={theta_c}, q={q_c}, n={n_c}, r_ser={r_ser_c}')
print(f'The parameters for no sub are: x0={x0_n}, y0={y0_n}, theta={theta_n}, q={q_n}, n={n_n}, r_ser={r_ser_n}')

# now we want to reconstruct the source with this parameter estimation
from source_reconstruct import reconstruct_source

no_sub_reconstruct = reconstruct_source(no_sub, profile_size=profile_size, pixel_width=pixel_width)
cdm_reconstruct = reconstruct_source(cdm, profile_size=profile_size, pixel_width=pixel_width)

# we want to estimate the parameters of the sersic profile for a reconstructed source
def sersic_reconstructed_source(x, y, reconstructed_source, profile_size, normalise=True):
    
    # check if reconstructed_source.shape[0] and reconstructed_source.shape[1] is equal to profile_size
    if reconstructed_source.shape[0] != profile_size or reconstructed_source.shape[1] != profile_size:
        raise ValueError(f"The shape of the image_profile must be equal to (profile_size, profile_size)=({profile_size, profile_size})")
    
    # estimate parameters using the function 'estimate_parameters'
    x0, y0, theta, q, n, r_ser = estimate_parameters(x, y, reconstructed_source, profile_size)
    
    sersic_parameters = {'x0': x0, 'y0': y0, 'theta': theta, 'q': q, 'n': n, 'r_ser': r_ser}
    
    # get the maximum intensity I0:
    I0 = reconstructed_source.max()
    
    # initialise array for storing Sersic profile intensities
    sersic_intensities = []
    
    # calculate the Sersic profile for all coordinates
    for x_coord in x:
        for y_coord in y:
            intensity = sersic_profile(x_coord, y_coord, x0, y0, theta, q, n, r_ser, I0=I0)
            sersic_intensities.append(intensity)
    
    # convert the list of intensities to a 2D array
    estimated_source = np.array(sersic_intensities).reshape(profile_size, profile_size)
    
    if normalise:
        estimated_source = estimated_source/estimated_source.max()
    
    return sersic_parameters, estimated_source   

# now we can get the estimated sersic parameters and estimated source profile
# of the lensed images
sersic_params_cdm, estimated_source_cdm = sersic_reconstructed_source(x, y, cdm_reconstruct, 
                                                                      profile_size=profile_size)

sersic_params_nosub, estimated_source_nosub = sersic_reconstructed_source(x, y, no_sub_reconstruct,
                                                                          profile_size=profile_size)

# function to plot the comparison between the gravitationally lensed image, the reconstructed source 
# and the reconstructed source assuming a sersic profile
def plot_comparison_sersic(image_profile, reconstructed_source, estimated_source, geometry, title='', 
                           cmap='inferno', colorbar=True, show_ticks=30, axis=False, vmin=None, vmax=None, 
                           min_angle_vision=-3.232, max_angle_vision=3.232):
    
    # prepare a list of dictionaries containing image data and titles for 
    # each image to be plotted
    images = [
        {'img': image_profile, 'cmap': cmap, 'title': 'Gravitationally Lensed Image'},
        {'img': reconstructed_source, 'cmap': cmap, 'title': 'Reconstructed Source Galaxy'},
        {'img': estimated_source, 'cmap': cmap, 'title': 'Sersic Estimation (Source)'},
        {'img': estimated_source - reconstructed_source, 'cmap': cmap, 'title': 'Residual'}]
      
    # get the centre coordinates of the image
    height, width = image_profile.shape[:2]
    center_x, center_y = width/2, height/2
    
    # create tick positions and labels
    x_tick_labels = np.array([int(100 * min_angle_vision)/100, 0, int(100 * max_angle_vision)/100])
    y_tick_labels = np.array([int(100 * min_angle_vision) / 100, 0, int(100 * max_angle_vision)/ 100])
    
    # create a 1 by 4 grid for subplots
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    
    for i, (ax, image) in enumerate(zip(axes, images)):
        # plot each image in the corresponding subplot
        ax.imshow(image['img'], cmap=image['cmap'], extent=[-center_x, center_x, -center_y, center_y])
        ax.set_title(image['title'], fontsize=10)
        
        # set the number of ticks on each axis and corresponding labels
        ax.set_xticks(np.linspace(-center_x, center_y, num=3))
        ax.set_xticklabels(x_tick_labels)
        
        ax.set_yticks(np.linspace(-center_x, center_y, num=3))
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel('x [arcsec]')
        
        # for the first subplot, add the ylabel and hide the y-axis for the rest
        if i == 0:
            ax.set_ylabel(' y [arcsec]')
        else:
            ax.yaxis.set_visible(False)
        
    # add the main title above the subplots
    plt.suptitle(title, fontweight='bold')
    
    # add additional label to mention the lens geometry
    fig.text(0.5, 0.85, geometry, ha='center', fontsize=30)
    plt.show()

print(f'The estimated sersic parameters for the reconstructed cdm profile are: {sersic_params_cdm}')
print(f'The sersic parameters for the reconstructed no substructure profile are: {sersic_params_nosub}')

# now we want to plot the comparison between the lensed image, the reconstructed source and
# the reconstructed source with the estimated sersic profile approximation 
# first for the lens with cdm substructure
plot_comparison_sersic(cdm, cdm_reconstruct, estimated_source_cdm, geometry='Cold Dark Matter')

# now for the lens with no dark matter substructure
plot_comparison_sersic(no_sub, no_sub_reconstruct, estimated_source_nosub, geometry='No Substructure')


