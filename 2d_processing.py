import cv2
import numpy as np

def empty(a):
    pass

# HSV threshold calibration
# returns HSV min and max thresholds
def hsv_trackbars(img):
    # default: 130 136 40 52 110 131
    cv2.namedWindow("TrackBars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TrackBars",  640, 200)
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
        mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2)) # make smaller
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

    cv2.namedWindow("Edge Detection TrackBars")
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


def main():
    file_name = "Caren_Graphene_50x_cropped.png"
    img = cv2.imread(file_name)

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







if __name__ == "__main__":
    main()