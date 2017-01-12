'''
Created on Dec 30, 2016

@author: iskandar
'''


import numpy as np;
import cv2;


cap = cv2.VideoCapture('../../resources/video/solidWhiteRight.mp4');

#R 191
#G 191
#B 0

#R 255
#G 255
#B 255

blueLower = np.array([0, 140, 200], dtype = "uint8")
blueUpper = np.array([255, 255, 255], dtype = "uint8")

gray = lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
gaussian_blur = lambda gray_frame, kernel: cv2.GaussianBlur(gray_frame, (kernel, kernel), 0);


while(True):
    (ret, frame) = cap.read();
    
    if not (ret):
        break;
    
    masked_frame = cv2.inRange(frame, blueLower, blueUpper);
    cv2.imshow('filter', masked_frame);

    masked_frame = gaussian_blur(masked_frame, 3);
    
    
    (_, contours, _) = cv2.findContours(masked_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    
    if(len(contours) > 0x0):
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0];
        
        rectangle = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
        cv2.drawContours(frame, [rectangle], -1, (0,255,0), 2);
    
    cv2.imshow('camera', frame);
    cv2.imshow('masked camera', masked_frame);
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break;
    
cap.release()
cv2.destroyAllWindows()