#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:19:40 2023

@author: ruby
"""

# Opening dark matter dataset
import numpy as np
import matplotlib.pyplot as plt

wdir = '/Users/ruby/Documents/Python Scripts/PhysicsInformedFeatures/'

# open the dataset
data_path = wdir+'Data/'

no_sub = np.load(data_path+'nosubstructure.npy')
cdm = np.load(data_path+'colddarkmatter.npy')
#print(f'The number of training samples is: {len(x_train)}.')
#print(f'The number of test samples is: {len(x_test)}.')

# create a 1x2 grid of subplots
#fig, axes = plt.subplots(1, 2, figsize=(8, 4))

images = [{'kind': 'No Substructure', 'img': no_sub},
          {'kind': 'Cold Dark Matter', 'img': cdm}]

# for i, (ax, image) in enumerate(zip(axes, images)):
#     # plot each type of lens
#     ax.imshow(image['img'], cmap='inferno')
#     ax.set_title(image['kind'])
#     ax.axis('off')
# plt.show()

profile_size = 64 #corresponds to 5 arcseconds

# now we want to reconstruct the source galaxy given the lensed image

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

def plot_image(matrix, title=None, cmap='inferno', colorbar=True, show_ticks=30,
               axis=False, vmin=None, vmax=None, min_angle_vision=-3.232, max_angle_vision=3.232):
    
    # calculate the center coordinates of the image
    height, width = matrix.shape[:2]
    center_x, center_y = width /2, height / 2
    
    # create tick positions and labels
    x_ticks = np.linspace(min_angle_vision, max_angle_vision, width+1)
    y_ticks = np.linspace(min_angle_vision, max_angle_vision, height+1)
    x_tick_labels = np.round(100 * x_ticks * center_x/75).astype(int)/100
    y_tick_labels = np.round(100 * y_ticks * center_y/75).astype(int)/100
    
    # plot the image with centered coordinates and updated ticks
    if vmax is None and vmin is None:
        plt.imshow(matrix, extent=[-center_x, center_x, -center_y, center_y], cmap=cmap)
    elif vmax is not None and vmin is None:
        plt.imshow(matrix, extent=[-center_x, center_x, -center_y, center_y], cmap=cmap, vmax=vmax)
    elif vmax is None and vmin is not None:
        plt.imshow(matrix, extent=[-center_x, center_x, -center_y, center_y], cmap=cmap, vmin=vmin)
    else:
        plt.imshow(matrix, extent=[-center_x, center_x, -center_y, center_y], cmap=cmap, vmin=vmin, vmax=vmax)
    
    # set the number of ticks on each axis
    plt.xticks(np.linspace(-center_x, center_y, width+1)[::show_ticks], x_tick_labels[::show_ticks])
    plt.yticks(np.linspace(-center_x, center_y, width+1)[::show_ticks], y_tick_labels[::show_ticks])
    
    if not axis:
        plt.axis('off')
    if title:
        plt.title(title)
    if colorbar:
        plt.colorbar()
    plt.show()
    
def reconstruct_source(image_profile, profile_size, pixel_width, normalise=True, k=1):
    # check if the image_profile[0] and image_profile.shape[1] is equal to profile_size
    if image_profile.shape[0] != profile_size or image_profile.shape[1] != profile_size:
        raise ValueError(f"The shape of the image_profile must be equal to (profile_size, profile_size)=({profile_size, profile_size})")
    
    # calculating the half of the profile size
    half_profile_size = int(profile_size/2)
    
    # initialise the array for the image coordinates
    image_coords = []
    
    # generate image coordinates based on the profile size and pixel width
    for x_index in range(-half_profile_size, half_profile_size):
        for y_index in range(-half_profile_size, half_profile_size):
            image_coords.append((x_index * pixel_width, y_index * pixel_width))
    
    # initialise the array for the reconstructed source
    reconstructed_source = np.zeros((profile_size, profile_size))
    
    # iterate over all coordinates in the lensed image
    for x_coord, y_coord in image_coords:
        # skip over center of image
        if x_coord != 0 or y_coord != 0:
            # calculate the new source coordinates using the lens equation
            # supposing singular isothermal ellipsoid
            x_source = (x_coord - k * x_coord / np.sqrt(x_coord**2 + y_coord**2))
            y_source = (y_coord - k * y_coord / np.sqrt(x_coord**2 + y_coord**2))
            
            # convert the source coordinates to index coordinates
            x_source_index = int(((x_source / pixel_width) + half_profile_size))
            y_source_index = int(((y_source / pixel_width) + half_profile_size))
            
            x_image_index = int((x_coord / pixel_width) + half_profile_size)
            y_image_index = int((y_coord / pixel_width) + half_profile_size)
            
            # update the pixel value in the reconstructed source
            if reconstructed_source[x_source_index][y_source_index] == 0:
                reconstructed_source[x_source_index][y_source_index] = image_profile[x_image_index][y_image_index]
            else:
                # the singular isothermal ellipsoid can have two images so we take the average
                reconstructed_source[x_source_index][y_source_index] = (reconstructed_source[x_source_index][y_source_index] +
                                                                        image_profile[x_image_index][y_image_index]) / 2
                
    if normalise:
        # normalise the reconstructed source
        reconstructed_source = reconstructed_source / reconstructed_source.max()
        
    return reconstructed_source

def plot_comparison(image_profile, reconstructed_source, geometry, title='', cmap='inferno', colorbar=True, show_ticks=30, 
                    axis=False, vmin=None, vmax=None, min_angle_vision=-3.232, max_angle_vision=3.232):
    
    # prepare a list of dictionaries containing image data and titles for 
    # each image to be plotted
    images = [
        {'img': image_profile, 'cmap': cmap, 'title': 'Gravitationally Lensed Image'},
        {'img': reconstructed_source, 'cmap': cmap, 'title': 'Reconstructed Source Galaxy'}]
      
    # get the centre coordinates of the image
    height, width = image_profile.shape[:2]
    center_x, center_y = width/2, height/2
    
    # create tick positions and labels
    x_tick_labels = np.array([int(100 * min_angle_vision)/100, 0, int(100 * max_angle_vision)/100])
    y_tick_labels = np.array([int(100 * min_angle_vision) / 100, 0, int(100 * max_angle_vision)/ 100])
    
    # create a 1 by 4 grid for subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    
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
    
    # add additional label to mention the lens geometry
    fig.text(0.5, 0.85, geometry, ha='center', fontsize=30)
    plt.show()

# create the coordinate grid for plotting
x, y, pixel_width = coords(profile_size=profile_size)

# reconstruct the source
reconstruct_nosub = reconstruct_source(image_profile=no_sub, profile_size=profile_size,
                                       pixel_width=pixel_width)

reconstruct_cdm = reconstruct_source(image_profile=cdm, profile_size=profile_size,
                                     pixel_width=pixel_width)

#plot_comparison(no_sub, reconstruct_nosub, geometry='No Substructure')
#plot_comparison(cdm, reconstruct_cdm, geometry='Cold Dark Matter')                           


#plt.imshow(no_sub-cdm, cmap='inferno')