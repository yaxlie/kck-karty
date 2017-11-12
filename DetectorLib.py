import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.09 * peri, True)

        return len(approx) == 4 or len(approx) == 5

            #(x, y, w, h) = cv2.boundingRect(approx)
            #ar = w / float(h)

    def getArea(self, image, contour):
        (x, y, w, h) = cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]
