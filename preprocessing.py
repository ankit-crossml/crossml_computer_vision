import cv2 
import numpy as np

def get_horizontal_lines(gray_scale): 

    length = np.array(gray_scale).shape[1]//100
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    horizontal_detect = cv2.erode(gray_scale, horizontal_kernel, iterations=3)
    hor_lines = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)

    return hor_lines

def get_vertical_lines(gray_scale): 
    length = np.array(gray_scale).shape[1]//100
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    vertical_detect = cv2.erode(gray_scale, vertical_kernel, iterations=3)
    ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)

    return ver_lines

def get_table_from_lines(gray, hor_lines, ver_lines):
    final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combine = cv2.addWeighted(ver_lines, 0.5, hor_lines, 0.5, 0.0)
    combine = cv2.erode(~combine, final, iterations=2)
    thresh, combine = cv2.threshold(combine,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    convert_xor = cv2.bitwise_xor(gray,combine)
    inverse = cv2.bitwise_not(convert_xor)

    return inverse, combine 

def preprocess(image): 

    cv2.imshow("Original Image",image)
    cv2.waitKey(0)
    # grayscale image 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # thresholding 
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # thresholcing using OTSU
    convert_bin,grey_scale = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # binary inverse 
    grey_scale = 255-grey_scale

    # getting horizontal lines 
    hor_lines = get_horizontal_lines(grey_scale)

    # getting vertical lines 
    ver_lines = get_vertical_lines(grey_scale)

    # combined image with tables 
    inverse, combined = get_table_from_lines(gray, hor_lines, ver_lines)

    cv2.imshow("Gray Scale", grey_scale)
    cv2.waitKey(0)

    cv2.imshow("horizontal lines", hor_lines)
    cv2.waitKey(0)

    cv2.imshow("Vertical Lines", ver_lines)
    cv2.waitKey(0)

    cv2.imshow("Combined", combined)
    cv2.waitKey(0)

    return gray, combined
    

