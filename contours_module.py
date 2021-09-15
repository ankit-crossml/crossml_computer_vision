import cv2 
import numpy as np


def draw_contour(image, contour, index=-1): 
    cont_im = cv2.drawContours(image.copy(), contour, -1,(0,255,0),3)

    cv2.imshow("Contours",cont_im)
    cv2.waitKey(0)


def get_contours(combined, image, draw = True):
    cont, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if draw: 
        draw_contour(image, cont)
    return cont, hierarchy


def get_contours_by_hierarchy(image, contours, hierarchy, h_value, crop = False ):
    im_copy = image.copy()
    hierarchy = hierarchy[0]
    crop_images = []
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x,y,w,h = cv2.boundingRect(currentContour)
        if currentHierarchy[3] == h_value and cv2.contourArea(currentContour)>2000: 
            # these are the innermost child components
            cv2.rectangle(im_copy,(x,y),(x+w,y+h),(0,0,255),3)
            if crop: 
                crop_images.append([x,y,w,h])


    cv2.imshow("Contours by Heirarchy", im_copy)
    cv2.waitKey(0)
    
    if crop: 
        for images in crop_images:
            x,y,w,h = images
            cv2.imshow("Crop Image", im_copy[y:y+h, x:x+w])
            cv2.waitKey(0)
    return im_copy  

    



