import numpy as np 
import cv2 
import os 
from glob import glob 

num_files = 0

def remove_salt_pepper_noise(image): 
    global num_files

    im = np.copy(image)

    median_blur = cv2.medianBlur(im, 3)

    cv2.imshow("De noised image",median_blur)
    
    num_files += 1 
    cv2.imwrite("saved_images/image_denoised" + str(num_files) + ".png", median_blur)
    
    k = cv2.waitKey(0)

    if k == 110:
        remove_salt_pepper_noise(median_blur)

    else: 
        return 0


if __name__ == "__main__":
    image_files = glob("images/*.png")

    for image_file in image_files: 
        image = cv2.imread(image_file)

        cv2.imshow("Original Image", image)
        cv2.waitKey(0)
        
        remove_salt_pepper_noise(image)