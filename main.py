import cv2 
import numpy as np
from preprocessing import * 
from contours_module import *


im = cv2.imread("0191273000.tif")

gray, combined = preprocess(im)

contours, hierarchy = get_contours(combined, im)

for h_value in [0,2,99]:
    crop = False 
    if h_value == 0: 
        crop = True
    get_contours_by_hierarchy(im, contours,hierarchy, h_value, crop)
# getting contours by heirarchy 





