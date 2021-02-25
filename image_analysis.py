import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors # for rgb to hsv
import scipy
from scipy import stats

# need some way to test this in parts: define each as a function
def nothing(x):
    print(x)

# use trackbars to change the thresholds of canny edge detection
def trackbars():
    img = cv2.imread("g1.jpg", 0)# 1 = color, 0 = grayscale
    img = cv2.resize(img, (1000, 1000))
    
    # trackbars
    cv2.namedWindow('image')
    cv2.createTrackbar('lowerThreshold', 'image', 0, 255, nothing)
    cv2.createTrackbar('upperThreshold', 'image', 0, 255, nothing)
    
    
    titles = ['Graphene image', 'Canny Graphene image']
    while(1):
        lower = cv2.getTrackbarPos('lowerThreshold', 'image')
        upper = cv2.getTrackbarPos('upperThreshold', 'image')
        canny = cv2.Canny(img, lower, upper)
        cv2.imshow('canny image', canny)
    
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

# repeat a similar process for color thresholding
def color_thresh():
    # read the image
    img = cv2.imread("g1.jpg", 1)
    img = cv2.resize(img, (1000, 1000))
    cv2.imshow('Color Image', img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
    
    # get the background color as mode of pixel values
    background_color = stats.mode(img)[0][0][0]
    # show a 100x100 square of what background color was chosen
    background = np.zeros((100, 100, 3), np.uint8)
    background[:] = background_color
    print("background_color =", background_color)
    print("background matrix[0][0] =", background[0][0])
    print("img[0][0] =", img[0][0])
    cv2.imshow('Background color', background)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
    
    # remove background color
    color_diff = 25 # allowable difference to background color to pixel that will be removed
    for row in range(len(img)):
        for col in range(len(img[0])):
            b = img[row][col][0]
            g = img[row][col][1]
            r = img[row][col][2]
            #print(row, col, "r, g, b =", r, g, b)
            b_diff = abs(int(b) - int(background_color[0]))
            g_diff = abs(int(g) - int(background_color[1]))
            r_diff = abs(int(r) - int(background_color[2]))
            if b_diff < color_diff and g_diff < color_diff and r_diff < color_diff:
                #print("Removing pixel ({}, {}) with b_diff, g_diff, r_diff = ({}, {}, {})".format(row, col, b_diff, g_diff, r_diff))
                img[row][col] = (0, 0, 0)
                sr, sc = row, col
            
    cv2.imshow('Updated image threshold = 25', img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()

# see function name...
def convert_to_hsv(img_rgb):
    # regularize the pixels to [0, 1]
    img_rgb = img_rgb / 255
    # convert to hsv
    img_hsv = colors.rgb_to_hsv(img_rgb)
    #print(img_hsv)
    return img_hsv

# input: hsv image
def show_hsv(img_hsv):
    cv2.imshow('HSV image', img_hsv)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()


def main():
    img = cv2.imread("g1.jpg", 1)
    img = cv2.resize(img, (1000, 1000))

    #trackbars()
    #color_thresh()

    # testing convert to hsv
    img = convert_to_hsv(img)
    show_hsv(img)

if __name__ == "__main__":
    main()
