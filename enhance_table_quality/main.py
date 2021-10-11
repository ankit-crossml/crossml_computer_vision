import numpy as np
import cv2
import os 
from glob import glob 

def find_discarded_indices(sorted_points, index_value, max_dist = 50, thresh_dist= 20):
    """Finds the discarded indices given an input array of points and parameters

    Args:
        sorted_points ([np.ndarray]): [Sorted array of shape (len,4), the input array from which we would like to remove the close lines]
        index_value ([int]): [1 for horizontal lines and 0 for vertical lines, used in sorting]
        max_length (int, optional): [Maximum lenth to stop search for a given line]. Defaults to 50.
        thresh (int, optional): [threshold value, line is discarded if it's distance to the line is less then thresh]. Defaults to 20.
    """
        # finding default indices in vertical lines 
    discarded_indices = [] 
    
    for i in range(len(sorted_points)-1): 
        value_1 = sorted_points[i][index_value]

        for j in range(i+1,len(sorted_points)):
            if j in discarded_indices: 
                continue 
            value_2 = sorted_points[j][index_value]
            dist = value_2 - value_1
            if dist > max_dist: 
                break 
            elif dist < thresh_dist:
                discarded_indices.append(j)

    return discarded_indices

def remove_discarded_lines(sorted_lines, discarded_indices): 
    """Remove the discarded indices from a sorted array of points

    Args:
        sorted_lines (np.ndarray): [numpy array of shape (num_points,4)]
        discarded_indices ([type]): [list containing the indeces which are to be removed]
    """
    selected_points = []

    for i in range(len(sorted_lines)): 
        if not i in discarded_indices: 
            selected_points.append(sorted_lines[i])
    return np.array(selected_points)


def remove_close_lines(points,img): 
    """Takes an input of points and returns points deleting those who are nearby

    Steps involded - 
        - clipping points in len of the image (max of the width and height)
        - Separating the horizontal and vertical lines 
        - Sorting the horizontal and vertical lines 
        - Run the function find_discarded_indices to remove the discarded indices 
        - Remove the discarded indices 
        - Merge horizontal and vertical lines 
        - return points

    Args:
        points ([np.ndarray]): [np array of shape (num_lines,4) where each instance 
        represents an array of points (x1,y1,x2,y2)]
        
    """
    # clip points 
    points_clipped = np.clip(points, 0, max(img.shape))

    # select horizontal and vertical lines
    lines_horizontal = points_clipped[points_clipped[:,0] == 0]
    lines_vertical = points_clipped[points_clipped[:,3] == 0]

    # sorting horizontal and vertical lines
    lines_horizontal_sorted = lines_horizontal[lines_horizontal[:,1].argsort()]
    lines_vertical_sorted = lines_vertical[lines_vertical[:,0].argsort()]

    # finding discarded indices for horizontal and vertical lines 
    discarded_indices_horizontal = find_discarded_indices(lines_horizontal_sorted, 1)
    discarded_indices_vertical = find_discarded_indices(lines_vertical_sorted, 0)

    ##########################debugging###########################################
    print("Number of horizontal lines removed - ",len(discarded_indices_horizontal))
    print("Number of vertical lines removed - ", len(discarded_indices_vertical))

    # remove discarded horizontal and vertical lines 
    selected_lines_horizontal = remove_discarded_lines(lines_horizontal_sorted, discarded_indices_horizontal)
    selected_lines_vertical = remove_discarded_lines(lines_vertical_sorted, discarded_indices_vertical)


    print(selected_lines_horizontal.shape)
    print(selected_lines_vertical.shape)
    return np.concatenate((selected_lines_horizontal, selected_lines_vertical))
    

def get_lines(img):
  image = img.copy()
  if image.ndim == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else: 
    gray=image
  edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

  # plt.figure(figsize = (12,8))
  # plt.title("edges")
  # plt.imshow(edges, cmap="gray")

  # Run HoughLines using a rho accuracy of 1 pixel
  # Theta accuracy of np.pi / 180
  # Our line threshold is set to 200 (number of points on line)
  lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
  # We iterate through each line and convert it to the format
  # required by cv.lines
  points = []
  for line in lines:
    rho, theta = line[0]
    
    a = np.cos(theta)
    b = np.sin(theta)
    
    x0 = a * rho    
    y0 = b * rho
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))

    # if x1 < 0: 
    #     x1 = 0 
    #     x2 = image.shape[1]
    # if y2 < 0: 
    # y1= 0
    # y2 = image.shape[0]


    thresh = 5 
    if np.absolute(x2-x1) > thresh and np.absolute(y2-y1) > thresh: 
        continue
    else:
        points.append([x1,y1,x2,y2])

  filtered_points = remove_close_lines(points, image)
  for line in filtered_points: 
      x1,y1,x2,y2 = line
      cv2.line(image, (x1,y1), (x2,y2), (0,0,0),3)  
  return image


if __name__ == '__main__': 
    files = glob("images/*.png")

    for image_path in files:
        image = cv2.imread(image_path)

        cv2.imshow("Original Image", image)

        

        enhanced_image = get_lines(image)

        cv2.imshow("Enhanced Image",enhanced_image)
        cv2.waitKey(0)
