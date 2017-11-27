import cv2
import time
import imutils
import preProc
import DetectorLib
import Sliders
import math

cap = cv2.VideoCapture(0)

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videostream = cv2.VideoWriter('output.avi', fourcc, 20.0, (IM_WIDTH, IM_HEIGHT))
time.sleep(1)

sliders = Sliders.Sliders()
preProc = preProc.PreProc()

cam_quit = 0

while cam_quit == 0:
    ret, frame = cap.read()

    thresh = preProc.preprocess_image(frame, sliders.gamma, sliders.contrast, sliders.mean, True)

    cardsDetectod = DetectorLib.CardsDetector(frame, thresh)

    cards, c = cardsDetectod.getRotatedCards()

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

        # tutaj kopiuję wycinek z oryginalnego obrazu (wnętrze wykrytego konturu)
    i = -1
    for card in cards:
        i+=1
        #cv2.imshow(str(i), card)
        text = (preProc.cutCard(frame, card, sliders.gamma, sliders.contrast, sliders.mean))
        textMark = (preProc.cutMark(frame, card, sliders.gamma, sliders.contrast, sliders.mean))

        M = cv2.moments(c[i])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.putText(cardsDetectod.image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        cv2.putText(cardsDetectod.image, textMark, (cX, cY + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 2)

    cv2.imshow("Image", cardsDetectod.image)
    cv2.resizeWindow('Image', 800, 600)
    cv2.moveWindow("Image", 0, 0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

cv2.destroyAllWindows()

# todo main jest do posprzątania, ten środek można wrzucić do jakiejś nowej klasy
# todo wykrywanie konturów powinno być robione na górnej połowie białego fragmentu obrazu (inaczej np. wychwytuje dłoń trzymającą kartę
# todo poprawić wykrywanie więcej kart w dłoni, częściowo nakładających się

# --- przydatne linki
#   https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
#   http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
