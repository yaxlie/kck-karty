import numpy as np
import cv2
import time

from PIL import ImageEnhance, Image
from skimage.measure import compare_ssim


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

def cutCard(image, card,g, c, m):
    global x, y
    cardCoppy = card
    posBegin = np.zeros((4,2), dtype = "float32")
    cutX = 0
    cutY = 0
    h = card.shape[0] -1
    w = card.shape[1] -1
    licznik = 0
    pom = []
    maxWidth = 200
    maxHeight = 300
    card = preprocess_image(card, g, c, m, debug=False)
    cutX = card[int(w/4-1):int(w/4),:]
    for item in cutX[0]:
        licznik = licznik + 1
        if item == 255:
            pom.append(licznik)
    if len(pom) > 0:
        cutX = pom[0]
        card = card[:,int(cutX):]
        cardCoppy = cardCoppy[:,int(cutX):];
    licznik = 0
    pom = [];
    cutY = card[:,int(h/4-1):int(h/4)]
    for item in cutY:
        licznik = licznik + 1
        if item == 255:
            pom.append(licznik)
    if len(pom) > 0:
        cutY = pom[0]
        card = card[int(cutY):,:]
        cardCoppy = cardCoppy[int(cutY):,:]
    licznik = 0
    pom = [];
    
    #funkcja filtrujaca ruch karty
    startX = 5
    startY = 0

    finishX = startX
    finishY = startY

    corner = cardCoppy[startX:startX+25, startX:startX+24]

    W = 75
    H = 125

    mark = corner
    posBegin = np.float32([[0,0],[14,0],[14,20],[0,20]])
    posEnd = np.array([[0,0],[75-1,0],[75-1,125-1],[0, 125-1]], np.float32)
    M = cv2.getPerspectiveTransform(posBegin,posEnd)
    mark = cv2.warpPerspective(mark, M, (75, 125))
    mark = preprocess_image(mark, 8, c, 180, debug=False)

    corner = cv2.warpPerspective(corner, M, (75, 125))

    # corner = preprocess_image(corner, g, c, m, debug=False)
    corner = preprocess_image(corner, 4, c, 180, debug=False)


    for x in range(0,74):
        find = False
        for y in range(0, 124):
            if corner[y,x] == 0:
                finishX = x
                find = True
                break
        if find:
            break

    for y in range(0, 124):
        find = False
        for x in range(0, 74):
            if corner[y,x] == 0:
                finishY = y
                find = True
                break
        if find:
            break

    # corner = corner[finishY:finishY + H, finishX:finishX +W]
    corner = corner[finishY:finishY+H, finishX:finishX+W]
    corner = cv2.resize(corner, (75, 125))

    # cv2.imshow("debug1", mark)
    cv2.imshow("debug2",corner)
    return findCards(corner)


def findCards(mark):
    tablica = []
    piksele = []
    wynik = []
    dd = []
    licznik = 0
    k = 255 - cv2.imread("Ace.png",0)
    tablica.append(255 - k)
    piksele.append(int(np.sum(k / 255)))

    k = 255 -  cv2.imread("9.png", 0)
    tablica.append(255 - k)
    piksele.append(int(np.sum(k / 255)))

    k = 255 -  cv2.imread("10.png", 0)
    tablica.append(255 - k)
    piksele.append(int(np.sum(k / 255)))

    k = 255 -  cv2.imread("J.png", 0)
    tablica.append(255 - k)
    piksele.append(int(np.sum(k / 255)))

    k = 255 -  cv2.imread("Q.png", 0)
    tablica.append(255 - k)
    piksele.append(int(np.sum(k / 255)))

    k = 255 -  cv2.imread("K.png", 0)
    tablica.append(255 - k)
    piksele.append(int(np.sum(k / 255)))


    for pos in tablica:
        # (score, diff) = compare_ssim(mark, pos, full=True)
        # diff = (diff * 255).astype("uint8")
        # print("SSIM: {}".format(score), " ")
        t2 = 255-mark
        wynik.append(pos+t2)
        # cv2.imshow("Twynik", pos)

    print("\n")
    for i in range(0, len(wynik)):
        dd.append(np.sum(wynik[i]/255)/piksele[i])
        cv2.imshow("wynik", wynik[i])

    print(dd, "\n")
    wyn = np.argmin(dd)

    if(wyn == 0):
        return "Ace"
    elif(wyn == 1):
        return "10"
    elif(wyn == 2):
        return "9"
    elif(wyn == 3):
        return "J"
    elif(wyn == 4):
        return "Q"
    elif(wyn == 5):
        return "K"
    else:
        return "Unknown"


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

def oldThresh(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    mean = np.mean(blur )
    if(mean > 255):
        mean=255
    retval, thresh = cv2.threshold(blur, mean, 255, cv2.THRESH_BINARY)

    return thresh
