'''
Created on Dec 30, 2016

@author: iskandar
'''

import numpy as np;
import matplotlib.pyplot as plt;
import cv2;

gray = lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
gaussian_blur = lambda gray_frame, parameters: cv2.GaussianBlur(gray_frame, (parameters['kernel'], parameters['kernel']), 0);
canny = lambda blur_frame, parameters: cv2.Canny(blur_frame, parameters['low_threshold'], parameters['high_threshold']);
grey = lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
polygon = lambda: np.array([[(0, 540), (430, 300), (530, 300), (900, 540)]], dtype=np.int32);




parameters = {
        'rho': 2,
        'theta': np.pi / 180,
        'threshold': 30,
        'min_line_len': 10,
        'max_line_gap': 1,
        'kernel': 5,
        'low_threshold': 50,
        'high_threshold': 150,
        'low_color_threshold': [0, 140, 200],
        'high_color_threshold': [255, 255, 255]
        };



frame = cv2.imread('../../resources/images/test_images/solidWhiteRight.jpg');

blueLower = np.array(parameters['low_color_threshold'], dtype = "uint8")
blueUpper = np.array(parameters['high_color_threshold'], dtype = "uint8")

cv2.fillPoly(mask, vertices, ignore_mask_color);


filtered_frame = cv2.inRange(frame, blueLower, blueUpper);

plt.imshow(filtered_frame);
plt.show();
filtered_frame = gaussian_blur(filtered_frame, 3);

(_, contours, _) = cv2.findContours(filtered_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
if(len(contours) > 0x0):
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0];
        
    rectangle = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
    cv2.drawContours(frame, [rectangle], -1, (0,255,0), 2);
    
cv2.waitKey(0);
    
frame.release();
cv2.destroyAllWindows();