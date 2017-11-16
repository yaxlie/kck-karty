import numpy as np
import cv2
import time

from PIL import ImageEnhance, Image


def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    img = adjust_gamma(image, 0.8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img)
    contrast = ImageEnhance.Contrast(pil_im)
    contrast = contrast.enhance(10)
    img = np.array(contrast)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 1)
    mean = np.mean(img[::2] ** 1.22)

    retval, thresh = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)


    # ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

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
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        value = 128 + factor * (c - 128)
        return max(0, min(255, value))
    return img.point(contrast)

    rank_diff = int(np.sum(marged)/255)
    print(rank_diff)
    return "Ace"
