'''
Created on Dec 14, 2016

@author: iskandar
'''
import sys
print(sys.version)

import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;
import cv2;

image = mpimg.imread('../../resources/images/exit-ramp.jpg');

grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);

kernel = 3;

gfilter = cv2.GaussianBlur(grayscale, (kernel, kernel), 0);

#cv2.imshow("grayscale", gfilter);


low_threshold = 55;
high_threshold = 165;

gfilter_canny = cv2.Canny(gfilter, low_threshold, high_threshold);


cv2.imshow("Canny", gfilter_canny);

cv2.waitKey(0);
