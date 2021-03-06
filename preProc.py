import numpy as np
import cv2
import time

from PIL import ImageEnhance, Image
from skimage.measure import moments, moments_central, moments_normalized

class PreProc:
    def __init__(self):
        self.tablicaCards = []
        self.pikseleCards = []

        k = cv2.imread("A.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("ABD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("AD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("AB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("ABB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))

        k = cv2.imread("9.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("9BD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("9D.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("9B.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("9BB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))

        k = cv2.imread("10.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("10BD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("10D.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("10B.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("10BB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))

        k = cv2.imread("J.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("JBD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("JD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("JB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("JBB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))

        k = cv2.imread("Q.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("QBD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("QD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("QB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("QBB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))

        k = cv2.imread("K.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("KBD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("KD.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("KB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))
        k = cv2.imread("KBB.png", 0)
        self.tablicaCards.append(k)
        self.pikseleCards.append(int(np.sum(k / 255)))

        pass

    def preprocess_image(self, image, g, c, m, debug=False):
        """Returns a grayed, blurred, and adaptively thresholded camera image."""

        img = self.adjust_gamma(image, g)

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

    def cutCard(self, image, card,g, c, m):
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
        card = self.preprocess_image(card, g, c, m, debug=False)
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
        startX = 0
        startY = 3

        finishX = startX
        finishY = startY

        corner = cardCoppy[startY:startY+25, startX:startX+24]

        W = 75
        H = 125

        mark = corner
        posBegin = np.float32([[0,0],[14,0],[14,20],[0,20]])
        posEnd = np.array([[0,0],[75-1,0],[75-1,125-1],[0, 125-1]], np.float32)
        M = cv2.getPerspectiveTransform(posBegin,posEnd)
        mark = cv2.warpPerspective(mark, M, (75, 125))
        mark = self.preprocess_image(mark, 8, c, 180, debug=False)

        corner = cv2.warpPerspective(corner, M, (75, 125))

        # corner = preprocess_image(corner, g, c, m, debug=False)
        corner = self.preprocess_image(corner, 3, c, 180, debug=False)


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
        #new function
        # k = cv2.imread("Jnew.png",0)
        # corner2 = cv2.resize(corner, (120, 200))
        # res = cv2.matchTemplate(k, corner2, cv2.TM_CCOEFF_NORMED)
        #
        # threshold = 0.4
        # loc = np.where(res >= threshold)
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(corner2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        # cv2.imshow("ss",corner2)
        # cv2.imshow("debug1", mark)
        cv2.imshow("mark",corner)
        cv2.moveWindow("mark", 75, 0)
        return self.findCards(corner)


    def findCards(self, mark):
        wynik = []
        dd = []
        licznik = 0

        for pos in self.tablicaCards:
            wynik.append(cv2.absdiff(mark,pos))
        cv2.imshow("roznice", wynik[5])
        cv2.moveWindow("roznice", 0, 0)
        for i in range(0, len(wynik)):
            dd.append(np.sum(wynik[i]/255))

        print(dd, "\n")
        dd.append(2500)
        wyn = np.argmin(dd)

        if(wyn == 0):
            return "Ace"
        elif(wyn == 1):
            return "Ace"
        elif(wyn == 2):
            return "Ace"
        elif(wyn == 3):
            return "Ace"
        elif(wyn == 4):
            return "Ace"

        elif(wyn == 5):
            return "9"
        elif(wyn == 6):
            return "9"
        elif(wyn == 7):
            return "9"
        elif(wyn == 8):
            return "9"
        elif(wyn == 9):
            return "9"

        elif(wyn == 10):
            return "10"
        elif(wyn == 11):
            return "10"
        elif(wyn == 12):
            return "10"
        elif(wyn == 13):
            return "10"
        elif(wyn == 14):
            return "10"

        elif(wyn == 15):
            return "J"
        elif(wyn == 16):
            return "J"
        elif(wyn == 17):
            return "J"
        elif(wyn == 18):
            return "J"
        elif(wyn == 19):
            return "J"

        elif(wyn == 20):
            return "Q"
        elif(wyn == 21):
            return "Q"
        elif(wyn == 22):
            return "Q"
        elif(wyn == 23):
            return "Q"
        elif(wyn == 24):
            return "Q"

        elif(wyn == 25):
            return "K"
        elif(wyn == 26):
            return "K"
        elif(wyn == 27):
            return "K"
        elif(wyn == 28):
            return "K"
        elif(wyn == 29):
            return "K"

        else:
            return "Unknown"


    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def change_contrast(self, img, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            value = 128 + factor * (c - 128)
            return max(0, min(255, value))
        return img.point(contrast)

    def oldThresh(self, image):
        """Returns a grayed, blurred, and adaptively thresholded camera image."""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        mean = np.mean(blur )
        if(mean > 255):
            mean=255
        retval, thresh = cv2.threshold(blur, mean, 255, cv2.THRESH_BINARY)

        return thresh

    def huMoment(self, img):
        h = 1 - img
        m = moments(h)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = moments_central(h, cr, cc)
        return mu


    def findMarks(self, mark):
        tablica = []
        piksele = []
        wynik = []
        dd = []
        licznik = 0

        k = cv2.imread("Kier.png", 0)
        tablica.append(k)
        piksele.append(int(np.sum(k / 255)))
        k = cv2.imread("Karo.png", 0)
        tablica.append(k)
        piksele.append(int(np.sum(k / 255)))
        k = cv2.imread("Trefl.png", 0)
        tablica.append(k)
        piksele.append(int(np.sum(k / 255)))
        k = cv2.imread("Pik.png", 0)
        tablica.append(k)
        piksele.append(int(np.sum(k / 255)))


        for pos in tablica:
            wynik.append(cv2.absdiff(mark, pos))
        cv2.imshow("rZnak", wynik[3])
        cv2.moveWindow("rZnak", 0, 300)
        for i in range(0, len(wynik)):
            dd.append(np.sum(wynik[i] / 255))

        print(dd, "\n")
        dd.append(2500)
        wyn = np.argmin(dd)

        if (wyn == 0):
            return "Kier"
        elif (wyn == 1):
            return "Karo"
        elif (wyn == 2):
            return "Trefl"
        elif (wyn == 3):
            return "Pik"

        else:
            return "Znaczek"

    def cutMark(self, image, card,g, c, m):
        hw = card[1]
        hh = card[0]
        global x, y
        cardCoppy = card
        h = card.shape[0] - 1
        w = card.shape[1] - 1
        licznik = 0
        pom = []
        card = self.preprocess_image(card, g, c, m, debug=False)
        cutX = card[int(w / 4 - 1):int(w / 4), :]
        for item in cutX[0]:
            licznik = licznik + 1
            if item == 255:
                pom.append(licznik)
        if len(pom) > 0:
            cutX = pom[0]
            card = card[:, int(cutX):]
            cardCoppy = cardCoppy[:, int(cutX):];
        licznik = 0
        pom = [];
        cutY = card[:, int(h / 4 - 1):int(h / 4)]
        for item in cutY:
            licznik = licznik + 1
            if item == 255:
                pom.append(licznik)
        if len(pom) > 0:
            cutY = pom[0]
            card = card[int(cutY):, :]
            cardCoppy = cardCoppy[int(cutY):, :]
        pom = [];

        startX = 0
        startY = 25

        finishX = startX
        finishY = startY

        corner = cardCoppy[startY:, startX:]
        h = corner.shape[0] -1
        w = corner.shape[1] -1
        W = 75
        H = 125

        mark = corner
        posBegin = np.float32([[int(0.03*w), 0], [int(0.18*w), 0], [int(0.18*w), int(0.1*h)], [0, 0.1*h]])
        posEnd = np.array([[0, 0], [75 - 1, 0], [75 - 1, 125 - 1], [0, 125 - 1]], np.float32)
        M = cv2.getPerspectiveTransform(posBegin, posEnd)
        mark = cv2.warpPerspective(mark, M, (75, 125))
        mark = self.preprocess_image(mark, 8, c, 180, debug=False)

        corner = cv2.warpPerspective(corner, M, (75, 125))

        corner = self.preprocess_image(corner, 3, c, 180, debug=False)
        cv2.imshow("testtttttt",corner)
        for x in range(0, 74):
            find = False
            for y in range(0, 124):
                if corner[y, x] == 0:
                    finishX = x
                    find = True
                    break
            if find:
                break

        for y in range(0, 124):
            find = False
            for x in range(0, 74):
                if corner[y, x] == 0:
                    finishY = y
                    find = True
                    break
            if find:
                break

        corner = corner[finishY:finishY + H, finishX:finishX + W]
        corner = cv2.resize(corner, (75, 125))

        cv2.imshow("znak", corner)
        cv2.moveWindow("znak", 75, 300)


        return self.findMarks(corner)

