import numpy as np
import cv2
import time

from PIL import ImageEnhance, Image


def preprocess_image(image, g, c, m, debug=False):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    img = adjust_gamma(image, g)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img)
    contrast = ImageEnhance.Contrast(pil_im)
    contrast = contrast.enhance(c)
    img = np.array(contrast)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 1)
    #mean = np.mean(img[::2] ** m)


    retval, thresh = cv2.threshold(img, m, 255, cv2.THRESH_BINARY)

    if debug:
        cv2.namedWindow('Grey', cv2.WINDOW_NORMAL)
        cv2.imshow("Grey", img)
        cv2.resizeWindow('Grey', 200, 200)
        cv2.moveWindow("Grey", 800, 0)

        cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)
        cv2.imshow("Thresh", thresh)
        cv2.resizeWindow('Thresh', 200, 200)
        cv2.moveWindow("Thresh", 800, 200)

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
    #cv2.imshow("debug",warp)
    return warp


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

