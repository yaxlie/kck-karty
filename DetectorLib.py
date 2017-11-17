import cv2
import imutils


class CardsDetector:
    def __init__(self, image, thresh):
        self.image = image.copy()
        self.imageOrigin = image.copy()
        self.thresh = thresh
        pass

    def detect(self, c):
        if len(c) > 100:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.10 * peri, True)

            return len(approx) == 4 or len(approx) == 5
        else:
            return False

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
        #wyświetlanie do debugowania
        if(debug):
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.imshow("Image", self.image)
            cv2.resizeWindow('Image', 800, 600)
            cv2.moveWindow("Image", 0, 0)
        return cards