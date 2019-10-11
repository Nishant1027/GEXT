import numpy as np
import cv2

def func(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(100,100))
    # downsize it to reduce processing time
    #cv2.imshow("original",frame)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #print(frame.shape)
    #tuned settings
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")

    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)

    # apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    lowerBoundary = np.array([170,80,30],dtype="uint8")
    upperBoundary = np.array([180,255,250],dtype="uint8")

    skinMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask2,0.5,0.0)
    #print(skinMask.flatten())
    #print(skinMask.shape)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    skin = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)
    cv2.imshow("masked",skin) # Everything apart from skin is shown to be black
    cv2.waitKey(10000)
    return skin
        
func(r'C:\Users\EMINACK\Desktop\GEXT\DATASET\A\001.jpg')
