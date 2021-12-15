
import cv2
import pytesseract
import os
import numpy as np
path="D:/Python/Rep1/Text-recog"
path=os.path.dirname(__file__)
path=path.replace("\\","/")


def recog(img):
    config = r"--oem 3 --psm 6 -l ukr --tessdata-dir {}/Tesseract-OCR/tessdata".format(path)
    pytesseract.pytesseract.tesseract_cmd = "{}/Tesseract-OCR/tesseract.exe".format(path)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img=final
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(img, config=config))

img = cv2.imread("{}/text.png".format(path))
recog(img)
cv2.imshow('Result', img)
cv2.waitKey(0)
