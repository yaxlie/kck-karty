import cv2
import time
import imutils
import preProc
import DetectorLib
import Sliders

cap = cv2.VideoCapture(0)

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videostream = cv2.VideoWriter('output.avi', fourcc, 20.0, (IM_WIDTH, IM_HEIGHT))
time.sleep(1)

sliders = Sliders.Sliders()

cam_quit = 0

while cam_quit == 0:

    ret, frame = cap.read()

    thresh = preProc.preprocess_image(frame, sliders.gamma, sliders.contrast, sliders.mean, True)

    cardsDetectod = DetectorLib.CardsDetector(frame, thresh)

    cards = cardsDetectod.detectCards(True)

        # tutaj kopiuję wycinek z oryginalnego obrazu (wnętrze wykrytego konturu)
    for card in cards:
         #cv2.imshow("karta", card)
        preProc.cutCard(frame, card, sliders.gamma, sliders.contrast, sliders.mean)

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
