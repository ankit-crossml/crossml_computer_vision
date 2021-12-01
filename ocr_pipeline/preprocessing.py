import cv2 
import numpy as np

def get_horizontal_lines(gray_scale): 
    # refer to the function below, it does the same but for horizontal lines
    length = np.array(gray_scale).shape[1]//100
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    horizontal_detect = cv2.erode(gray_scale, horizontal_kernel, iterations=3)
    hor_lines = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)

    return hor_lines

def get_vertical_lines(gray_scale): 
    """Given a gray scaled image, it finds the vertical lines usign a suitable kernel

    Args:
        gray_scale ([np.ndarray]): grayscale image

    Returns:
        [np.ndarray]: [image containing vertical lines]
    """
    length = np.array(gray_scale).shape[1]//100
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    vertical_detect = cv2.erode(gray_scale, vertical_kernel, iterations=3)
    ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)

    return ver_lines

def get_table_from_lines(gray, hor_lines, ver_lines):
    """From a given grayscale image, hor lines, and vertical lines, It finds the table 
        structure from the image. 


    Args:
        gray ([np.ndarray]): grayscale image
        hor_lines ([np.ndarray]): [an image containing the horizontal lines -> returned by get_hoirzontal_lines]
        ver_lines ([type]): [an image containing the vertical lines -> returned by get_vertical_lines]

    Returns:
        [type]: [description]
    """
    final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combine = cv2.addWeighted(ver_lines, 0.5, hor_lines, 0.5, 0.0)
    combine = cv2.erode(~combine, final, iterations=2)
    thresh, combine = cv2.threshold(combine,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    convert_xor = cv2.bitwise_xor(gray,combine)
    inverse = cv2.bitwise_not(convert_xor)

    return inverse, combine 

def preprocess(image, debug): 
    """Performs all the preprocessing operations on the given image

    Args:
        image ([np.ndarray]): [the image on which the preprocessing operations are to be performed]
        debug (bool, optional): [Whether or not we want to visualize the intermediate results]. Defaults to False.

    Returns:
        gray, combined: gray: grayscale of the input image 
                        combined: image containing the horizontal and vertical lines
    """


    # if image is colored, then convert it to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: 
        gray = image

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

    # for debbing purposes
    if debug: 
        cv2.imshow("Gray Scale", grey_scale)
        cv2.waitKey(0)

        cv2.imshow("horizontal lines", hor_lines)
        cv2.waitKey(0)

        cv2.imshow("Vertical Lines", ver_lines)
        cv2.waitKey(0)

        cv2.imshow("Combined", combined)
        cv2.waitKey(0)


    return gray, combined
    

