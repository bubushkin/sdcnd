'''
Created on Dec 18, 2016

@author: iskandar
'''

import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;
import numpy as np;
import cv2;
from numpy import dtype

import moviepy;
import imageio;

image = cv2.imread('../../resources/images/exit-ramp.jpg');

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);

kernel = 5;

blur_gray = cv2.GaussianBlur(gray, (kernel, kernel), 0);

low_threshold = 50;
high_threshold = 150;

edges = cv2.Canny(blur_gray, low_threshold, high_threshold);


mask = np.zeros_like(edges);
ignore_mask_color = 255;


imshape = image.shape;
#vertices = np.array([[(0, imshape[0]), (430, 300), (530, 300), (imshape[1], imshape[0])]], dtype=np.int32);
vertices = np.array([[(0, 540), (430, 300), (530, 300), (900, 540)]], dtype=np.int32);

cv2.fillPoly(mask, vertices, ignore_mask_color);
masked_edges = cv2.bitwise_and(edges, mask);

plt.imshow(masked_edges);


rho = 1;
theta = np.pi/180;
threshold = 30;
min_line_len = 10;
max_line_gap = 1;
line_image = np.copy(image) * 0;

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_len, max_line_gap);

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10);
        
color_edges = np.dstack((edges, edges, edges));

combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0);

#cv2.imshow("Lane lines", combo);
plt.imshow(combo);
plt.show();

cv2.waitKey(0);
