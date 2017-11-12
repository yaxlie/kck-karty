import cv2
import time
import imutils
import preProc
import DetectorLib

cap = cv2.VideoCapture(0)

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videostream = cv2.VideoWriter('output.avi', fourcc, 20.0, (IM_WIDTH, IM_HEIGHT))
time.sleep(1)

cam_quit = 0

while cam_quit == 0:

    ret, frame = cap.read()

    image = frame
    ratio = image.shape[0] / float(image.shape[0])

    thresh = preProc.preprocess_image(frame)
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = DetectorLib.ShapeDetector()

    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)


        if sd.detect(c):
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            if len(c)>100:
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.putText(image, "Karta", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

                #tutaj kopiuję wycinek z oryginalnego obrazu (wnętrze wykrytego konturu)
                cv2.imshow("karta", sd.getArea(image, c))

        #wyświetlanie do debugowania
        cv2.imshow("Image", image)
        cv2.imshow("thresh", thresh)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

cv2.destroyAllWindows()
videostream.release

# todo main jest do posprzątania, ten środek można wrzucić do jakiejś nowej klasy
# todo wykrywanie konturów powinno być robione na górnej połowie białego fragmentu obrazu (inaczej np. wychwytuje dłoń trzymającą kartę
# todo poprawić wykrywanie więcej kart w dłoni, częściowo nakładających się

# --- przydatne linki
#   https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
#   http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html