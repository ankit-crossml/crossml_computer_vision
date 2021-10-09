import cv2 
import os
import numpy as np
from glob import glob  
from skimage import restoration

def deblur_deconvolution(image): 
    psf = np.ones((5, 5, 3)) / 25

    restored_image = restoration.richardson_lucy(image, psf, 1)

    return restored_image

def deblur_sharpen(image):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)

    return sharpen

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


if __name__ == '__main__':
    files = glob("images/*.png")
    num_saved_files = 0
    for file in files: 
        image = cv2.imread(file)

        score = variance_of_laplacian(image)

        print("blur Score - ",score)

        cv2.imshow("Image",image)
        k = cv2.waitKey(0)

        # if d is pressed -> (deblur)
        while k == 100:
            # deblur = deblur_deconvolution(image)
            deblur = deblur_sharpen(image)
            score = variance_of_laplacian(deblur)
            print("Blue score after debluring - ",score)
            cv2.imshow("Deblur", deblur)
            num_saved_files+=1
            cv2.imwrite("output_images/deblured_" + str(num_saved_files) + ".png", deblur)
            k = cv2.waitKey(0)