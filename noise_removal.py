import os
import cv2
import glob
import math
import numpy as np
from time import time
from scipy.signal import convolve2d
from scipy.ndimage import interpolation as inter

# def increase_brightness(img, value=30):
#     print("bright")
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     lim = 255 - value
#     v[v > lim] = 255
#     v[v <= lim] += value
#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img

def estimate_noise(I):
  start = time()
  H, W = I.shape
  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]
  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
  #print("TIME TAKEN estimate noise ::", time()-start)
  return sigma
    
def noise_removal(img):
    start = time()
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    #print("TIME TAKEN noise removal ::", time()-start)
    return (img)

def postprocess(rotated_img):
        denoiseee = None

        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        blurr_index = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_index = estimate_noise(gray)
       
        #print("Blur Index ::", blurr_index)
        #print("Noise Index ::", noise_index)    
        if blurr_index <= 200 and noise_index >= 5:
            sharpen_kernel = np.array([[-1,-1,-1], [-1 ,9 ,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(gray,-2,sharpen_kernel)
            denoiseee = noise_removal(sharpen)
          
        elif blurr_index <200:
             sharpen_kernel = np.array([[-1,-1,-1], [-1 ,9 ,-1], [-1,-1,-1]])
             denoiseee = cv2.filter2D(gray,-2,sharpen_kernel)
           
        elif noise_index >= 5:
             sharpen_kernel = np.array([[-1,-1,-1], [-1 ,9 ,-1], [-1,-1,-1]])
             denoiseee1 = noise_removal(gray)
             denoiseee = cv2.filter2D(denoiseee1,-2,sharpen_kernel)
        
        else:
              denoiseee = rotated_img
        
        #out = increase_brightness(denoiseee)
        return denoiseee
           