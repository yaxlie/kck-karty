import numpy as np
import cv2
import time


def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    mean = np.mean(blur * 1)
    if(mean > 255):
        mean=255
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
    corner = warp[10:84, 10:32] 
    mark = corner[0:48,:]
    print(mark.shape)
    posBegin = np.float32([[0,5],[21,5],[21,45],[0,45]])
    posEnd = np.array([[0,0],[75-1,0],[75-1,125-1],[0, 125-1]], np.float32)
    M = cv2.getPerspectiveTransform(posBegin,posEnd)
    mark = cv2.warpPerspective(mark, M, (75, 125))
    mark = mark[0:125,0:75]
    mark = preprocess_image(mark);
    cv2.imshow("debug",mark)
    return findCards(mark)
    



def findCards(mark):
    machCard = 100000;
    img = cv2.imread("A.png",0)
    marged = cv2.absdiff(mark, img)
    rank_diff = int(np.sum(marged)/255)
    print(rank_diff)
    return "Ace"
