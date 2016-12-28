'''
Created on Dec 25, 2016

@author: iskandar
'''


import numpy as np
import cv2
import random;




def get_dims(image): return {'x': image.shape[1], 'y': image.shape[0]};

def get_polygon():

    poly_region = {
            'left_bottom': [0, 539],
            'right_bottom': [960, 539],
            'apex': [470, 315]
            };
    print(poly_region['left_bottom'][1]);
    poly_fit = {
        'left': np.polyfit((poly_region['left_bottom'][0], poly_region['apex'][0]), (poly_region['left_bottom'][1], poly_region['apex'][1]), 1),
        'right': np.polyfit((poly_region['right_bottom'][0], poly_region['apex'][0]), (poly_region['right_bottom'][1], poly_region['apex'][1]), 1),
        'bottom': np.polyfit((poly_region['left_bottom'][0], poly_region['right_bottom'][0]), (poly_region['left_bottom'][1], poly_region['right_bottom'][1]), 1)
        }

    return poly_fit;


def get_polygon_region_limit(dims, polygon):

    XX, YY = np.meshgrid(np.arange(0, dims['x']), np.arange(0, dims['y']));     
    return (YY > (XX * polygon['left'][0] + polygon['left'][1])) & (YY > (XX * polygon['right'][0] + polygon['right'][1])) & (YY < (XX* polygon['bottom'][0] + polygon['bottom'][1]));

if __name__ == '__main__':
    
    
    red = green = blue = 200;
    rgb = [red, green, blue];
    

    cap = cv2.VideoCapture('../../resources/video/solidWhiteRight.mp4')

    while(cap.isOpened()):
        ret, frame = cap.read()
    
        #gframe = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2GRAY)
                
        binary_frame = np.copy(frame);
        frame_copy = np.copy(frame);
        
        threshold = (frame[:,:,0] < rgb[0])  | (frame[:,:,1] < rgb[1])  | (frame[:,:,2] < rgb[2]);
        
        binary_frame[threshold] = [0, 0, 0];
        
        poly = get_polygon_region_limit(get_dims(frame), get_polygon());
        
        frame_copy[~threshold & poly] = [0, 255, 0];
        
        cv2.imshow('frame',frame_copy)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


