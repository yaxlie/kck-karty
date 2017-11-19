import cv2
import imutils
import numpy as np
import math

import preProc


class CardsDetector:
    def __init__(self, image, thresh):
        self.image = image.copy()
        self.imageOrigin = image.copy()
        self.thresh = thresh
        pass

    def detect(self, c):
        if len(c) > 40:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.12 * peri, True)

            return (len(approx) == 4
                    #or len(approx) == 5
             )

    def getArea(self, image, contour):
        (x, y, w, h) = cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]

    def detectCards(self, debug = False):
        ratio = self.image.shape[0] / float(self.image.shape[0])
        # shape detector
        cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cards = []
        contours = []

        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)

            if  self.detect(c):

                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(self.image, [c], -1, (0, 255, 0), 2)
                cv2.putText(self.image, "Karta", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                cards.append(self.getArea(self.imageOrigin, c))
                contours.append(c)
        #wyświetlanie do debugowania
        if(debug):
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.imshow("Image", self.image)
            cv2.resizeWindow('Image', 800, 600)
            cv2.moveWindow("Image", 0, 0)
        return cards,contours

    def rotateCard(self, c, card, debug = False):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.12 * peri, True)

        myradians = math.atan2(approx[0][0][1]-approx[1][0][1], approx[0][0][0]-approx[1][0][0])
        angle = math.degrees(myradians)
        #print(angle)

        if abs(approx[0][0][1] - approx[1][0][1]) < abs(approx[0][0][0] - approx[1][0][0]):
            card = self.rotateImage(card, angle)
        elif abs(approx[0][0][1] - approx[1][0][1]) > abs(approx[0][0][0] - approx[1][0][0]):
            card = self.rotateImage(card, angle + 90)

        if (debug):
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image', 800, 600)
            i = 0
            for a in approx:
                i+=1
                #print(a[0][0], " ", a[0][1])
                cv2.putText(self.image, str(i), (a[0][0], a[0][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            cv2.imshow("Image2", self.image)
            cv2.moveWindow("Image2", 0, 0)

        return card

    def getRotatedCards(self, debug = True):
        cards,contours = self.detectCards(debug)
        roatedCards =[]
        for i in range(0,len(cards)):
            roatedCards.append(self.rotateCard(contours[i], cards[i]))
        return roatedCards

    def rotateImage(self, image, angle):
        center = tuple(np.array(image.shape[0:2]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, image.shape[0:2], flags=cv2.INTER_LINEAR)



