import numpy as np
import cv2
import time


def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    mean = np.mean(blur[::2] ** 1.1)
    retval, thresh = cv2.threshold(blur, mean, 255, cv2.THRESH_BINARY)

    return thresh

def cutCard(image, card):
    
    posBegin = np.zeros((4,2), dtype = "float32")
    
    h = card.shape[0] -1
    w = card.shape[1] -1
    
    maxWidth = 200
    maxHeight = 300
    #if good oriented
    if(h >= w):
        posBegin = np.float32([[0,0],[w,0],[w,h],[0,h]])

    #if card horizontal
    if(w > h):
        posBegin = np.float32([[0,h],[0,0],[w,0],[w,h]])
    
    posEnd = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(posBegin,posEnd)
    
    warp = cv2.warpPerspective(card, M, (maxWidth, maxHeight))
    cv2.imshow("debug",warp)
    return warp
