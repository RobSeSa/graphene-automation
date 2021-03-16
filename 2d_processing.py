import cv2
import numpy as np
import matplotlib.pyplot as plt

AREA_THRESHOLD = 100

def empty(a):
    pass

# HSV threshold calibration
# returns HSV min and max thresholds
def hsv_trackbars(img):
    # default: 130 136 40 52 110 131
    cv2.namedWindow("TrackBars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TrackBars",  840, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 130, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 136, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 40, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 52, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 110, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 131, 255, empty)

    # convert to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # repeat while moving trackbars
    while True:

        # get the trackbar values
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars") 
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper) # lower and upper limit on HSV image
        # gives us filtered out image of this color
        # white is pixels w value 1, black is pixels w value 0

        # use this mask to create a new image
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        # combine the images
        img1 = np.concatenate((img, imgHSV), axis=1)
        all_img = np.concatenate((img1, imgResult), axis=1)

        cv2.imshow("HSV Threshold Mask", all_img)
        mask = cv2.resize(mask, (mask.shape[1], mask.shape[0])) # make smaller
        cv2.imshow("Mask", mask)

        # repeat every 1 ms and break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
       
    return (h_min, h_max, s_min, s_max, v_min, v_max), imgResult

# create trackbars to set thresholds for the canny edge detection
# expects RGB image to conver tto grayscale
def canny_trackbar(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0) # blue the image slightly using a 7,7 mask

    cv2.namedWindow("Edge Detection TrackBars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edge Detection TrackBars", 640, 240)
    cv2.createTrackbar("Canny Min", "Edge Detection TrackBars", 0, 255, empty)
    cv2.createTrackbar("Canny Max", "Edge Detection TrackBars", 255, 255, empty)

    while True:
        c_min = cv2.getTrackbarPos("Canny Min", "Edge Detection TrackBars") 
        c_max = cv2.getTrackbarPos("Canny Max", "Edge Detection TrackBars")
        print(c_min, c_max)
        imgCanny = cv2.Canny(img, c_min, c_max) # edge detection algorithm to be applied
        cv2.imshow("Canny Image", imgCanny)
        # repeat every 1 ms and break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    return imgCanny

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # method for finding contours; try others
    print("Found", len(contours), "contours")
    for cnt in contours: # contours saved in contours
        # find the area
        area = cv2.contourArea(cnt)
        
        # draw the contour
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3) # -1 means all of the contours

        if area>AREA_THRESHOLD:
            peri = cv2.arcLength(cnt, True) # true is saying the arc is closed

            # get the points of corners in polygons
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True) # epsilon (2nd parameter) is accuracy

            # draw a bounding box
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imgContour, "Area:"+ str(area),
                        ((x+w), y+h), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 255, 255), 1)




def main():
    file_name = "Caren_Graphene_50x_cropped.png"
    #file_name = "graphene_close.jpg"
    original_img = cv2.imread(file_name)
    #print("original shape:", original_img.shape)
    #original_img = cv2.resize(original_img, (original_img.shape[1]//4, original_img.shape[0]//4))
    img = original_img.copy()

    # calibrate the HSV trackbars 
    if True:
        (h_min, h_max, s_min, s_max, v_min, v_max), img = hsv_trackbars(img)
        print("\nCalibrated HSV Thresholds:\nh_min:", h_min, "h_max:", h_max, "s_min:", s_min, "s_max:", s_max, "v_min:", v_min, "v_max:", v_max)
        # show calibrated image
        cv2.imshow("Calibrated HSV image:", img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()

    # apply edge detection
    imgCanny = canny_trackbar(img)
    print(imgCanny.shape)

    # draw bounding boxes on this edge image
    imgContour = img.copy()
    getContours(imgCanny, imgContour)

    all_img = np.concatenate((original_img, imgContour), axis=1)
    cv2.imshow("Original + Contour Image", all_img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()

    # horizontal line cuts to check entropy
    '''
    img_w_lines = original_img.copy()
    cv2.line(img_w_lines, (0, 200), (img_w_lines.shape[1], 200), (0, 0, 0),1) # red line of thickness 1
    cv2.line(img_w_lines, (0, 400), (img_w_lines.shape[1], 400), (0, 0, 0),1) # red line of thickness 1
    cv2.line(img_w_lines, (0, 600), (img_w_lines.shape[1], 600), (0, 0, 0),1) # red line of thickness 1
    cv2.imshow("Line cuts", img_w_lines)
    cv2.waitKey(0)

    print("Line cut:")
    cut1 = original_img[200][:]
    cut2 = original_img[400][:]
    cut3 = original_img[600][:]
    plt.plot(cut1)
    plt.show()
    plt.plot(cut2)
    plt.show()
    plt.plot(cut3)
    plt.show()
    '''








if __name__ == "__main__":
    main()
