import cv2
import os
import matplotlib.pyplot as plt


def func():
    face_cascade = cv2.CascadeClassifier('face_detector.xml')
    picCount = 0
    index = 1
    margin = 20
    for subdir, dirs, files in os.walk('vikivikianga'):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                img = cv2.imread(filepath)
                faces = face_cascade.detectMultiScale(img, 1.25, 7)
                picCount += 1
                for (x, y, w, h) in faces:
                    x = x - margin if margin < x else 0
                    y = y - margin if margin < y else 0
                    crop_img = img[y:y + h + margin, x:x + w + margin]
                    writeFileName = 'faces/V' + str(index) + ".jpg"
                    res = cv2.imwrite(writeFileName, crop_img)
                    index += 1
                print("Wrote " + str(len(faces)) + " images from:", picCount)


# func()
