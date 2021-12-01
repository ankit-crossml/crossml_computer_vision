import cv2 
import numpy as np


def draw_contour(image, contour, index=-1): 
    # draw a particular contour on a given image
    cont_im = cv2.drawContours(image.copy(), contour, -1,(0,255,0),3)

    cv2.imshow("Contours",cont_im)
    cv2.waitKey(0)


def get_contours(combined, image, debug= False):
    """Given an image (combined), containing the vertical and horizontal lines. Find the contours and plot them on image

    Args:
        combined ([np.ndarray]): [The image with the help of which contours are calculated]
        image ([type]): [The original image on which contours are drawn]
        debug (bool, optional): [Whether or not we want to draw the outputs]. Defaults to False.

    Returns:
        [type]: [description]
    """
    cont, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug: 
        draw_contour(image, cont)
    return cont, hierarchy


def get_contours_by_hierarchy(image, contours, hierarchy, h_value, crop = False, debug=False):
    """For all the contours in the image,  it selects the specific contours which belong to a given hierarchy
       The contours belonging to the given hierarchy are then cropped and saved in a list which is then returned by the function.

    Args:
        image ([np.ndarray]): [the image to be worked on]
        contours ([type]): [the countours object]
        hierarchy ([type]): [the heirarchy information of the contours]
        h_value ([type]): [the hierarchy value we want to extract]
        crop (bool, optional): [If we want to get the cropped images or not]. Defaults to False.
        debug (bool, optional): [to print the intermediate results for debugging]. Defaults to False.

    Returns:
        [list]: [list of selected images (cropped images) that we obtained for a particular heirarchy]
    """
    im_copy = image.copy()
    hierarchy = hierarchy[0]
    crop_images = []
    selected_images = []

    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x,y,w,h = cv2.boundingRect(currentContour)
        if currentHierarchy[3] == h_value and cv2.contourArea(currentContour)>20000: 
            # these are the innermost child components
            cv2.rectangle(im_copy,(x,y),(x+w,y+h),(0,0,255),3)
            if crop: 
                crop_images.append([x,y,w,h])

    if debug: 
        cv2.imshow("Contours by Heirarchy", im_copy)
        cv2.waitKey(0)
    
    if crop: 
        for images in crop_images:
            x,y,w,h = images
            cropped_image = im_copy[y:y+h, x:x+w]
            selected_images.append(cropped_image)
            if debug:
                cv2.imshow("Crop Image", cropped_image)
                cv2.waitKey(0)
    return selected_images  

    



