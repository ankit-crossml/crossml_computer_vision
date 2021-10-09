import numpy as np
import cv2 
import os 
from scipy.ndimage.interpolation import rotate
from glob import glob

def findScore(img, angle):
    """
    Generates a score for the binary image recieved dependent on the determined angle.\n
    Vars:\n
    - array <- numpy array of the label\n
    - angle <- predicted angle at which the image is rotated by\n
    Returns:\n
    - histogram of the image
    - score of potential angle
    """
    data = rotate(img, angle, reshape = False, order = 0)
    hist = np.sum(data, axis = 1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def skewCorrect(img):
    """
    Takes in a nparray and determines the skew angle of the text, then corrects the skew and returns the corrected image.\n
    Vars:\n
    - img <- numpy array of the label\n
    Returns:\n
    - Corrected image as a numpy array\n
    """
    #Crops down the skewImg to determine the skew angle
    img = cv2.resize(img, (0, 0), fx = 0.75, fy = 0.75)

    delta = 1
    limit = 45
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = findScore(img, angle)
        scores.append(score)
    bestScore = max(scores)
    bestAngle = angles[scores.index(bestScore)]
    rotated = rotate(img, bestAngle, reshape = False, order = 0)
    print("[INFO] angle: {:.3f}".format(bestAngle))
    #cv2.imshow("Original", img)
    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey(0)
    
    #Return img
    return rotated



def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 120, 255, 1)

    corners = cv2.goodFeaturesToTrack(canny,4,0.5,50)

    for corner in corners:
        x,y = corner.ravel()
        print(x)
        print(y)
        cv2.circle(image,(int(x),int(y)),5,(36,255,12),-1)

    cv2.imshow('canny', canny)
    cv2.imshow('image', image)
    cv2.waitKey(0)

if __name__=="__main__":
    files = glob('images/*.png')

    for file in files: 
        
        image = cv2.imread(file)

        corrected = skewCorrect(image)



        cv2.imshow("Image",image)

        cv2.imshow("Corrected", corrected)

        detect_corners(image)
        cv2.waitKey(0)