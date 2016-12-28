'''
Created on Dec 12, 2016

@author: iskandar
'''

import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;

import numpy as np;


image = mpimg.imread('../../resources/images/test.jpg');

print("Type:", type(image), ' dimensions: ', image.shape);


ysize = image.shape[0];
xsize = image.shape[1];

color_select = np.copy(image);

plt.imshow(image);

red_threshold = 200;
green_threshold = 200;
blue_threshold = 200;

rgb_thld = [red_threshold, green_threshold, blue_threshold];

threshold = (image[:,:,0] < rgb_thld[0]) \
            | (image[:,:,1] < rgb_thld[1]) \
            | (image[:,:,2] < rgb_thld[2]);
            
            
color_select[threshold] = [0, 0, 0];
plt.imshow(color_select); 
plt.show();

print();