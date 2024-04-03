import cv2
import numpy as np

spot = cv2.imread("spot.png", cv2.IMREAD_COLOR)
grey_spot = cv2.cvtColor(spot, cv2.COLOR_BGR2GRAY)

blur_grey_spot = cv2.blur(grey_spot, (3,3))
detected_circles = cv2.HoughCircles(blur_grey_spot, cv2.HOUGH_GRADIENT, 1, 20, param1= 50, param2= 30, minRadius= 1, maxRadius= 50)

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for i  in detected_circles[0, :]:
        a,b,r = i[0], i[1], i[2]
        cv2.circle(spot, (a,b), r, (244,144,111), 2)
        cv2.circle(spot, (a,b), 1, (111,144,244), 3)
        cv2.imshow("Detected circles", spot)
        cv2.waitKey(0)


cv2.destroyAllWindows()