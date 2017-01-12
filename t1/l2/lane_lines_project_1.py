'''
Created on Dec 28, 2016

@author: iskandar
'''

import numpy as np
import cv2


gray = lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
polygon = lambda: np.array([[(0, 540), (430, 300), (530, 300), (900, 540)]], dtype=np.int32);
dims = lambda image: {'x': image.shape[1], 'y': image.shape[0]};
hough_lines = lambda mask_region, parameters: cv2.HoughLinesP(mask_region, parameters['rho'], parameters['theta'], parameters['threshold'], np.array([]), parameters['min_line_len'], parameters['max_line_gap']);
mask_region = lambda edge_frame: cv2.bitwise_and(edge_frame, cv2.fillPoly(np.zeros_like(edge_frame), polygon(), 255));
canny = lambda blur_frame, parameters: cv2.Canny(blur_frame, parameters['low_threshold'], parameters['high_threshold']);
gaussian_blur = lambda gray_frame, parameters: cv2.GaussianBlur(gray_frame, (parameters['kernel'], parameters['kernel']), 0);

if(__name__ == '__main__'):
    
    
    parameters = {
        'rho': 2,
        'theta': np.pi / 180,
        'threshold': 30,
        'min_line_len': 10,
        'max_line_gap': 1,
        'kernel': 5,
        'low_threshold': 50,
        'high_threshold': 150
        };
    
    cap = cv2.VideoCapture('../../resources/video/solidWhiteRight.mp4')

    while(cap.isOpened()):
        ret, frame = cap.read()
        
        print(dims(frame));

        line_image = np.copy(frame) * 0;
        
        gray_frame = gray(frame);
        
        blur_gray_frame = gaussian_blur(gray_frame, parameters);
        
        edge_frame = canny(blur_gray_frame, parameters);
        
        lines_collection = hough_lines(mask_region(edge_frame), parameters);
        

        for line in lines_collection:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10);        

        color_edges = np.dstack((edge_frame, edge_frame, edge_frame));

        combo = cv2.addWeighted(frame, 0.5, line_image, 1, 0);
        
        cv2.imshow('frame', combo);
     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
        
