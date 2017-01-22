'''
Created on Jan 2, 2017

@author: iskandar
'''

import matplotlib.pyplot as plt;
import matplotlib.image as mpimg;
import numpy as np;
import cv2;
import math;
from scipy import stats;
from scipy.spatial.distance import euclidean

param = {
        'rho': 2,
        'theta': np.pi / 180,
        'threshold': 30,
        'min_line_len': 10,
        'max_line_gap': 1,
        'kernel': 3,
        'edge_low_threshold': 50,
        'edge_high_threshold': 150,
        'poly_left_bottom': [0, 539],
        'poly_right_bottom': [960, 539],
        'poly_apex': [470, 315],
        'low_color_threshold': np.array([0, 140, 200], dtype = "uint8"),
        'high_color_threshold': np.array([255, 255, 255], dtype = "uint8")
        };

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def poly_len(vertices):
    
    print(vertices);
    
    left_side = euclidean(vertices[0][0], vertices[0][1]);
    right_side = euclidean(vertices[0][2], vertices[0][3]);
    
    upper_base = euclidean(vertices[0][1], vertices[0][2]);
    lower_base = euclidean(vertices[0][0], vertices[0][3]);
    
    return (int(left_side), int(right_side), int(upper_base), int(lower_base));

def polygon(shape):
    """
    Returns trapezoid for the image as a region of interest.
    """
    
    x_ratio = 0.95;
    y_ratio = 0.62;
    x_mid_ratio = 0.3;
    
    magnitude_x = shape[1];
    magnitude_y = shape[0]; 
    
    vertices = np.array([
                          [
                            (int(magnitude_x * (1 - x_ratio)), magnitude_y), 
                            (int(magnitude_x / 3), int(magnitude_y * y_ratio)), 
                            (int(magnitude_x / 3) + int(magnitude_x * x_mid_ratio), int(magnitude_y * y_ratio)), 
                            (int(magnitude_x * x_ratio), magnitude_y)
                          ]
                         ], dtype=np.int32);


    return vertices;

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 255, 0], thickness=2, polygon=None):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    avg = 0x0;
    avg_left_slope = 0x0;
    avg_right_slope = 0x0;
    
    left_slope = []
    right_slope = []
    
    x_right_line = [];
    y_right_line = [];
    
    x_left_line  = [];
    y_left_line = [];
    
    right_line = [];
    left_line = [];
    x_center = img.shape[0x1] / 2;
    """
    y=mx+b
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            
            
            slope = ((y2-y1)/(x2-x1));
            
            if(x1 > x_center and x2 > x_center):
                right_line.append(line);
            else:
                left_line.append(line);
                
            if(slope < 0x0):
                left_slope.append(slope);
                
            elif(slope >= 0x0):
                right_slope.append(slope);
            
            for i in left_line:
                for x1,y1,x2,y2 in i:
                    x_left_line.append(x1);
                    x_left_line.append(x2);
                    y_left_line.append(y1);
                    y_left_line.append(y2);
            
            for i in right_line:
                for x1,y1,x2,y2 in i:
                    x_right_line.append(x1);
                    x_right_line.append(x2);
                    y_right_line.append(y1);
                    y_right_line.append(y2);


        
    #y = mx + b
    left_m, left_b, left_r, left_p, left_std_err = stats.linregress(x_left_line, y_left_line);


    right_m, right_b, right_r, right_p, right_std_err = stats.linregress(x_right_line, y_right_line);
    
    poly_left = polygon[0];
    poly_right = polygon[1];
    
    upper_base = polygon[2];
    lower_base = polygon[3];
    
    #based on: height = sqrt(side_1^2 - ((side_1^2 - side_2^2 + d^2)/2d)2)
    trapezoid_height = math.sqrt(math.pow(poly_left, 0x2) - 0x2 * ((math.pow(poly_left, 0x2) - math.pow(poly_right, 0x2) + math.pow(lower_base - upper_base, 0x2))/(0x2 * (lower_base - upper_base))));
    
    #x = (y - b)/m
    y1 = img.shape[0];
    y2 = int(trapezoid_height);
    
    right_x1 = int((y1 - right_b) / right_m)
    right_x2 = int((y2 - right_b) / right_m)
    
    left_x1 = int((y1 - left_b) / left_m)
    left_x2 = int((y2 - left_b) / left_m)    
                    
    avg_left_slope = np.mean(left_slope, dtype=np.float)
#    avg_right_slope = np.mean(right_slope, dtype=np.float);
    
    cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
                #cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, polygon):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, polygon=polygon)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def color_filter(image):
    
    filter = cv2.inRange(image, param['low_color_threshold'], param['high_color_threshold']);
    final_filter = cv2.bitwise_and(image, image, mask=filter);
    return final_filter;

def process(image):

    
    poly = polygon(image.shape);
    
    poly_length = poly_len(poly);
    
    filtered_image = color_filter(image);

    gray = grayscale(filtered_image);
    
    blur_image = gaussian_blur(gray, param['kernel']);

    edge_masked = canny(blur_image, param['edge_low_threshold'], param['edge_high_threshold'])
    
    masked = region_of_interest(edge_masked, poly)
    
    cv2.imshow('masked', masked);
    
    hough = hough_lines(masked, param['rho'], param['theta'], param['threshold'], param['min_line_len'], param['max_line_gap'], poly_length);
    
    cv2.imshow('hough', hough);
    
    res = weighted_img(hough, image);
    
    cv2.imshow('image', res);
    
    cv2.waitKey(0);
    cv2.destroyAllWindows();

if(__name__ == "__main__"):
    image = cv2.imread('../../resources/images/test_images/solidYellowCurve.jpg');
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape);
    
    process(image);
