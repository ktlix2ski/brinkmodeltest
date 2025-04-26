#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 07:57:52 2024

@author: msharp
"""

import requests
import os

# %% define a function to save the files

def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# %% get the urls of each velocity pair
base_url = "https://nsidc.org/apps/itslive-search/velocities/urls/"

# LeConte

year_start = "1985"
year_stop = "2023"

params = {
    "bbox": "-132.542,56.7394,-131.907,57.2838",
  # "bbox": "-132.590150,56.774054,-131.942744,57.176904",
  "start": year_start + "-01-01",
  "end": year_stop + "-12-31",
  "percent_valid_pixels": 20, # percent of valid glacier pixels
  "min_interval": 5, # days of separation between image pairs
  "max_interval": 100, # days of separation between image pairs
    # "compressed": True, #compressed is faster than transferring plain text.
  "version": 2, # version 1 requires an EDL access token header in the request
  # "mission": "S1",
}


# params = {
# "bbox": "-24.60,63.78,-15.11,66.30",
# "start": "2020-02-01",
# "end": "2021-05-01",
# "percent_valid_pixels": 20, # percent of valid glacier pixels
# "min_interval": 7, # days of separation between image pairs
# "max_interval": 100, # days of separation between image pairs
# # "compressed": True, # compressed is faster than transferring plain text.
# "version": 2
# }


# This will return a list of NetCDF files in AWS S3 that can be accessed
# in the cloud or externally
velocity_pairs = requests.get(base_url, params=params)

print('Found ' + str(len(velocity_pairs.content)) + ' velocity pairs')

# %% 

# folder to save the files
# nc_savefolder = '/Volumes/Sandisk4TB/PhD_MS/TARSAN/ITS_LIVE/netCDFs/velocity_pairs_TWITandTEIS/from_requests//max30daysep/nc/' + year + '/orig/'
# png_savefolder = '/Volumes/Sandisk4TB/PhD_MS/TARSAN/ITS_LIVE/netCDFs/velocity_pairs_TWITandTEIS/from_requests//max30daysep/png/' + year + '/orig/'
nc_savefolder = '/media/kayatroyer/KayaDrive/Thesis/Figures/Results/ItsLIVE_Velocity' + year_start + '_' + year_stop + '/orig/'
png_savefolder = '/media/kayatroyer/KayaDrive/Thesis/Figures/Results/ItsLIVE_Velocity' + year_start + '_' + year_stop + '/orig/'

# if the folders don't exist, make them!
if not os.path.exists(nc_savefolder):
    os.makedirs(nc_savefolder)
    
if not os.path.exists(png_savefolder):
    os.makedirs(png_savefolder)

# Check if the request was successful
if velocity_pairs.status_code == 200:
    
    # Check the response content type
    content_type = velocity_pairs.headers.get('Content-Type')
    if 'application/json' not in content_type:
        print(f"Unexpected content type: {content_type}")
    else:
        json_data = velocity_pairs.json()
        print(f"Found {len(json_data)} velocity pairs json")
        
    # Iterate over the URLs in velocity_pairs
    for url in velocity_pairs.json():
        
        # Extract URLs from the pair
        nc_url = url['url']
        png_url = url['browse_image']
        
        # Extract filenames from URLs
        nc_filename = nc_url.split('/')[-1]
        png_filename = png_url.split('/')[-1]
        
        # Decide where to save the files
        nc_save_path = os.path.join(nc_savefolder, nc_filename)
        png_save_path = os.path.join(png_savefolder, png_filename)
        
        # Download the files
        download_file(nc_url, nc_save_path)
        download_file(png_url, png_save_path)
        print(f"Downloaded {nc_filename} and {png_filename}")
else:
    print("Failed to fetch velocity pairs:", velocity_pairs.status_code)
    
