#!/usr/bin/python

# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap

# Get the path of the training set
parser = ap.ArgumentParser()
#parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = vars(parser.parse_args())

# Read the input image 
im = cv2.imread(args["image"])

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY_INV)

#print image 
cv2.namedWindow("Image grayscale", cv2.WINDOW_NORMAL)
cv2.imshow("Image grayscale", im_th)
cv2.waitKey()

# Find contours in the image
_,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour, method-1
rects = [None]*100
ctrCount=0
for ctr in ctrs:
    
    if cv2.contourArea(ctr) >= 500:
        rects[ctrCount] = cv2.boundingRect(ctr)
        ctrCount=ctrCount+1

print 'ctrCount = ', ctrCount

# Get rectangles contains each contour, method-2
#rects = [cv2.boundingRect(ctr) for ctr in ctrs]

newImageCount = 0
# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles around digits on original image
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    
    #--- Digit Separating ---
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    
    # Resize the image and save image
    try:
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        #roi = cv2.dilate(roi, (3, 3))
    
        name = 'image'+str(newImageCount)+'.jpg'
        cv2.imwrite(name, roi)
        newImageCount = newImageCount+1
        #cv2.namedWindow("ROIs", cv2.WINDOW_NORMAL)
        #cv2.imshow("ROIs", roi)
        #cv2.waitKey()
    except:
        print 'Error'


#Display Original Image and Rectangle around digits
cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
