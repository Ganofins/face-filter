import cv2
import sys
from PIL import Image

class FaceFilter:

    casc_path = "haarcascade_frontalface_default.xml"

    def __init__(self,img_name):
        self.img_name = img_name
        self.face_cascade = cv2.CascadeClassifier(self.casc_path)

    def remove_bg(self):
        filter_mask = Image.open(self.img_name)
        filter_mask = filter_mask.convert("RGBA")
        datas = filter_mask.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255,255,255,0))
            else:
                newData.append(item)
        filter_mask.putdata(newData)
        filter_mask.save("new_filter.png","PNG")

    def face_detection(self):
        self.remove_bg
        cv2.namedWindow("WebCam")
        vc = cv2.VideoCapture(0)
        filter_mask = cv2.imread("new_filter.png")
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
                #flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                frame[y:y+h, x:x+w] = cv2.resize(filter_mask,(w,h))
            
            cv2.imshow("WebCam", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

        cv2.destroyWindow("WebCam")
        vc.release()


face_filter = FaceFilter(sys.argv[1])
face_filter.face_detection()