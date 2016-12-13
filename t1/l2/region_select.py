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

region_select = np.copy(image);

left_bottom = [0, 539];
right_bottom = [960, 539];
apex = [500, 345];

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1);
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1);
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1);

XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize));

region_threshold = (YY > (XX*fit_left[0] + fit_left[1])) & (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] + fit_bottom[1]));

region_select[region_threshold] = [255, 0, 0];

plt.imshow(region_select);
print();
 
