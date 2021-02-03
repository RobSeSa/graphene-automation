import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def nothing(x):
    print(x)

img = cv2.imread("g1.jpg", 0)# 1 = color, 0 = grayscale

# trackbars
cv2.namedWindow('image')
cv2.createTrackbar('lowerThreshold', 'image', 0, 255, nothing)
cv2.createTrackbar('upperThreshold', 'image', 0, 255, nothing)


titles = ['Graphene image', 'Canny Graphene image']
while(1):
    lower = cv2.getTrackbarPos('lowerThreshold', 'image')
    upper = cv2.getTrackbarPos('upperThreshold', 'image')
    canny = cv2.Canny(img, lower, upper)
    small = cv2.resize(canny, (1000, 1000))
    cv2.imshow('canny image', small)

    #images = [img, canny]
    #print("Showing images")
    #for i in range(2):
    #    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    #    plt.title(titles[i])
    #    plt.xticks([]),plt.yticks([])

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
