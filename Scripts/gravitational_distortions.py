#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:26:38 2023

@author: ruby
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
# Opening dark matter dataset
import os
import glob

# estimating gravitational distortions caused by lensing 

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

x, y, pixel_width = coords(profile_size=profile_size)

# import functions from other modules to reconstruct the source galaxy and
# approximate the parameters of the source assuming a sersic profile
from source_reconstruct import reconstruct_source
from sersic_estimation import sersic_bn_constant, sersic_profile
from sersic_estimation import sersic_reconstructed_source

# to compute the distortions in gravitational potential due to lensing, we need to 
# calculate the derivatives of the effective sersic radius with respect to the 
# x- and y-coordinates in the image plane 

# To compute the derivatives, we need a function that can analytically compute
# the gradient of a complicated function using fifth order finite difference
def finite_difference(matrix):
    
    # the step used to calculate the gradient
    h = pixel_width
    
    # compute the forward difference for all but the last four rows
    forward_diff = (-25 * matrix[:-4] + 48 * matrix[1:-3] - 36 * matrix[2:-2] + 16 * matrix[3:-1] - 3 * matrix[4:])/ (12 * h)
    # now compute the backward difference (reverse the minus signs of the forward difference)
    backward_diff = (25 * matrix[-4:] - 48 * matrix[-5:-1] + 36 * matrix[-6:-2] - 16 * matrix[-7:-3] + 3 * matrix[-8:-4])/ (12 * h)
    
    # concatenate the forward and backward differences to get the gradient for the whole matrix
    gradient = np.concatenate((forward_diff, backward_diff), axis=0)

    return gradient

# now we need to compute the derivatives in both x- and y-directions using the
# forward difference scheme we implemented above in the x-direction and the 
# transponse of the forward difference scheme in the y-direction.
def compute_grad(matrix):
    
    # compute the gradient in the x-direction
    grad_x = finite_difference(matrix)
    
    # compute the gradient in the y-direction
    # To do this, we have to transpose the matrix to switch the x and y axes
    # then we can compute the gradient and transpose the result
    grad_y = finite_difference(matrix.T).T

    return grad_x, grad_y

# now we want a function to estimate the distortions in gravitational potential
# Given a lensed image and a set of estimated sersic parameters, we can compute
# a distortion map and its associated sersic parameters --> the sersic parameters
# describe the light distribution of the source galaxy
def estimate_distortions(image_profile, sersic_params, profile_size, normalise=True, quantile=0.985):
    
    # check if image_profile.shape[0] and image_profile.shape[1] are equal to profile_size
    if image_profile.shape[0] != profile_size or image_profile.shape[1] != profile_size:
        raise ValueError(f"The shape of the image_profile must be equal to (profile_size, profile_size)=({profile_size, profile_size})")

    # extract required Sersic parameters from the input dictionary
    n = sersic_params['n']
    r_ser = sersic_params['r_ser']
    
    # calculate the maximum pixel value in the image profile
    I0 = image_profile.max()
    
    # assign the image profile to I
    I = image_profile
    
    # calculate the Sersic bn constant using the Sersic index, n
    bn = sersic_bn_constant(n)
    
    # calculate the Sersic radius, R, based on the image profile and the Sersic parameters
    R = (r_ser * ((1/bn) * np.log(I0/I)) ** n) ** 2
    
    # compute the first derivatives of R in the x- and y-directions
    Rx, Ry = compute_grad(R)
    
    # compute the second derivatives of R in the x-direction
    Rx2, Rxy = compute_grad(Rx)
    
    # compute the second derivatives of R in the y-direction
    Ryx, Ry2 = compute_grad(Ry)
    
    # calculate the average of the mixed second derivatives
    gravitational_distortions = (Rxy + Ryx) / 2
    
    # check if post_process=True
    if not normalise:
        # return the mixed partial derivative without post processing
        return gravitational_distortions
    else:
        # avoid taking the maximum value for the normalisation due to high values
        # near the singularities where the potential diverges
        norm = np.quantile(gravitational_distortions, quantile)
        
        # normalise the mixed partial derivative using the quantile value
        gravitational_distortions = gravitational_distortions / norm
        
        # use the hyperbolic tanh to avoid singularities and see other distortions
        gravitational_distortions = np.tanh(gravitational_distortions)
        
        # use the absolute value to avoid numerical issues that could appear due
        # to the finite difference scheme where we have divergence (also discretisation)
        gravitational_distortions = np.abs(gravitational_distortions)
        
        # now return the normalised convergence map
        return gravitational_distortions


# now let's run through the process of reconstructing the source, estimating the
# source assuming a sersic profile and estimating the parameters of that sersic profile
# we'll start by reconstructing the source
cdm_reconstruct = reconstruct_source(cdm, profile_size=profile_size, pixel_width=pixel_width)
nosub_reconstruct = reconstruct_source(no_sub, profile_size=profile_size, pixel_width=pixel_width)

# now we'll estimate the sersic parameters and sersic profile of the source
sersic_params_cdm, estimated_source_cdm = sersic_reconstructed_source(x, y, cdm_reconstruct,
                                                                      profile_size=profile_size)

sersic_params_nosub, estimated_source_nosub = sersic_reconstructed_source(x, y, nosub_reconstruct,
                                                                          profile_size=profile_size)

# and we'll get the distortions in gravitational potential
gravitational_distortions_cdm = estimate_distortions(cdm, sersic_params_cdm, profile_size=profile_size)
gravitational_distortions_nosub = estimate_distortions(no_sub, sersic_params_nosub, profile_size=profile_size)


# now, let's write a function to plot the original gravitationally lensed image,
# the reconstructed source, the estimated source profile assuming a sersic profile
# and the gravitational distortions along with the estimated sersic parameters
def plot_features(image_profile, reconstructed_source, estimated_source_profile, gravitational_distortions,
                 sersic_params, cmap='inferno', cmap_distortions='hot', figsize=(15, 5), title='',
                 min_angle_vision=-3.232, max_angle_vision=3.232):
    
    # prepare a list of dictionaries containing image data and titles for 
    # each image to be plotted
    images = [
        {'img': image_profile, 'cmap': cmap, 'title': 'Gravitationally Lensed Image'},
        {'img': reconstructed_source, 'cmap': cmap, 'title': 'Reconstructed Source Galaxy'},
        {'img': estimated_source_profile, 'cmap': cmap, 'title': 'Estimated Sersic Galaxy'},
        {'img': gravitational_distortions, 'cmap': cmap_distortions, 'title': 'Distortions in Gravitational Potential'}]
    
    # get the centre coordinates of the image
    height, width = image_profile.shape[:2]
    center_x, center_y = width/2, height/2
    
    # create tick positions and labels
    x_tick_labels = np.array([int(100 * min_angle_vision)/100, 0, int(100 * max_angle_vision)/100])
    y_tick_labels = np.array([int(100 * min_angle_vision) / 100, 0, int(100 * max_angle_vision)/ 100])
    
    # create a 1 by 4 grid for subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    for i, (ax, image) in enumerate(zip(axes, images)):
        # plot each image in the corresponding subplot
        ax.imshow(image['img'], cmap=image['cmap'], extent=[-center_x, center_x, -center_y, center_y])
        ax.set_title(image['title'])
        
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
    
    # add the additional label to explain the Sersic parameters
    fig.text(0.5, 0.88, 'Estimated Sersic Galaxy Parameters', ha='center', fontsize=10)
    
    # extract the parameters from the dictionary
    x0 = sersic_params['x0']
    # add a minus sign due to the inversion of the y-axis in the normalisation
    y0 = -sersic_params['y0']
    theta = sersic_params['theta']
    q = sersic_params['q']
    n = sersic_params['n']
    r_ser = sersic_params['r_ser']
    
    # format each variable to 2 decimal places
    x0_str = f"{x0:.2f}"
    y0_str = f"{y0:.2f}"
    theta_str = f"{theta:.2f}"
    q_str = f"{q:.2f}"
    n_str = f"{n:.2f}"
    r_ser_str = f"{r_ser:.2f}"
    
    # create the string
    string = fr"$x_0 = {x0_str}, y_) = {y0_str}, \theta = {theta_str}, q = {q_str}, n = {n_str}, R_\mathrm{{ser}} = {r_ser_str}$"
    
    # additional label below the title
    fig.text(0.5, 0.85, string, ha='center', fontsize=8)
    
    plt.show()

# now let's plot the cold dark matter distortions
plot_features(cdm, cdm_reconstruct, estimated_source_cdm, gravitational_distortions_cdm, sersic_params_cdm, 
              title='Cold Dark Matter')

plot_features(no_sub, nosub_reconstruct, estimated_source_nosub, gravitational_distortions_nosub,
              sersic_params_nosub, title='No Dark Matter')

