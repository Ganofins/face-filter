import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cv2.namedWindow("WebCam")
vc = cv2.VideoCapture(0)

mask = cv2.imread("mask.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w] = cv2.resize(mask,(w,h))
    
    cv2.imshow("WebCam", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("WebCam")
vc.release()