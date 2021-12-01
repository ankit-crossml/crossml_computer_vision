import cv2 
import os
import numpy as np

from contours_module import * 
from preprocessing import * 

# DEBUG=False
# doc_number = 1
# hurry=True

def save_images_in_disk(list_images, saving_location='saved_images/'):
    """ 
    Given a list of images. It saves the output to the local system. 
    Required when we get cropped portions from the document, and need to save them.

    The format of the name of the image -> "31.png" -> 3rd document 1st page
    Args:
        list_images ([list]): [List of image ]
    """
    # representing the document number
    global doc_number

    # representing the page number
    file_number = 1

    for image in list_images:
        # cv2.imwrite((os.path.join(os.getcwd(), saving_location, str(doc_number) + str(file_number) + ".png")), image)
        path = "saved_images/" + str(doc_number) + str(file_number) + ".png"
        print(path)
        print("-----------------------")
        cv2.imwrite(path, image)
        print("Image saved ")
        file_number += 1 
        
    doc_number += 1 

def process_image(im, debug=False):
    """
    Processing the first page of the document
    Args:
        im ([np.ndarray]): the image (the first page of the tiff document)
        debug ([bool]): whether or not we want to print the intermediate steps

    Returns:
        [list]: list of all the cropped images (mainly containing tables)
    """
    # do the preprcessing operations
    gray, combined = preprocess(im, debug=False)

    # get contours
    contours, hierarchy = get_contours(combined, im, debug)
    saved_images_total = []

    # filter contours for our need 
    for h_value in [0,2,99]:
        crop = False 
        if h_value == 0: 
            crop = True
        saved_images = get_contours_by_hierarchy(im, contours,hierarchy, h_value, crop, debug=False)
        saved_images_total.extend(saved_images)

    return saved_images_total

# getting contours by heirarchy 

# # loads all the tiff files and stores in a list
# docs = open_multi_docs()

# # iterate over all the docs in the list
# for doc in docs: 
    
#     # get the first page
#     first_page = doc[1][0]  
#     # process the first page
#     saved_images = process_first_page(first_page, debug = DEBUG, hurry=hurry)
#     # print("Number of images = ",len(saved_images))
#     # save the images obtained
#     save_images_in_disk(saved_images)


#     # check if there are more then 1 pages in the current doc
#     if check_pages(doc) > 1:
#         # get the second page and repeat the same process
#         second_page = doc[1][1]
#         saved_images = process_first_page(second_page, debug = DEBUG, hurry=hurry)
#         print("Number of imagees = ",len(saved_images))
#         save_images_in_disk(saved_images)




