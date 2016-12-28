'''
Created on Dec 28, 2016

@author: iskandar
'''

import numpy as np
import cv2


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


def get_dims(image): return {'x': image.shape[1], 'y': image.shape[0]};

def get_polygon():
    #TODO: do not use hardcoded values;
    return np.array([[(0, 540), (430, 300), (530, 300), (900, 540)]], dtype=np.int32);

def get_mask_region(edge_frame):
    
    ignore_mask_color = 255;
    return cv2.bitwise_and(edge_frame, cv2.fillPoly(np.zeros_like(edge_frame), get_polygon(), ignore_mask_color));


if(__name__ == '__main__'):
    
    cap = cv2.VideoCapture('../../resources/video/solidWhiteRight.mp4')

    while(cap.isOpened()):
        ret, frame = cap.read()

        line_image = np.copy(frame) * 0;
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
        
        blur_gray_frame = cv2.GaussianBlur(gray_frame, (parameters['kernel'], parameters['kernel']), 0);
        
        edge_frame = cv2.Canny(blur_gray_frame, parameters['low_threshold'], parameters['high_threshold']);
        
        lines_collection = cv2.HoughLinesP(get_mask_region(edge_frame), parameters['rho'], parameters['theta'], parameters['threshold'], np.array([]), parameters['min_line_len'], parameters['max_line_gap']);
        
        for line in lines_collection:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10);        

        color_edges = np.dstack((edge_frame, edge_frame, edge_frame));

        combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0);
        
        cv2.imshow('frame', combo);
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
        
