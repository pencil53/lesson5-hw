import cv2
import numpy as np

spot = cv2.imread("spot.png", 0)

#initializing simpleblockDector class and setting/filtering parameters
blob_detector = cv2.SimpleBlobDetector_Params()

#set area for filtering parameters
blob_detector.filterByArea = True
blob_detector.minArea = 100

#set circulatory filtering parameters
blob_detector.filterByCircularity = True
blob_detector.minCircularity = 0.2

#set convexy filtering parameters
blob_detector.filterByConvexity = True
blob_detector.minConvexity = 0.9

#set inertia filtering parameters
blob_detector.filterByInertia = True
blob_detector.minInertiaRatio = 0.01

#create detector with parameters
detector = cv2.SimpleBlobDetector_create(blob_detector)

#detect blobs
blobs = detector.detect(spot)

#draw blobs in red circle
blank = np.zeros((1,1))
redblob = cv2.drawKeypoints(spot, blobs, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

numblobs = len(blobs)
text = "there were {} blobs in this image".format(numblobs)
cv2.putText(redblob, text, (20,520), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,100), 2)

cv2.imshow("blob detection", redblob)
cv2.waitKey(0)

cv2.destroyAllWindows()