import numpy as np
import cv2 as cv

haar_cascade=cv.CascadeClassifier('haar_face.xml')

people=['ben','madona']

"""features=np.load('features.npy')
labels=np.load('labels.npy')
"""
face_recog=cv.face.LBPHFaceRecognizer_create()
face_recog.read('face_trained.yml')

img=cv.imread(r'C:\Users\kashi\Downloads\OPENCV\faces\validation\val_madona\4.jpg')

grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('grey',grey)

faces_rect=haar_cascade.detectMultiScale(grey,1.1,4)

for (x,y,w,h) in faces_rect:

    faces_roi=grey[y:y+h,x:x+w]

    label,confidence= face_recog.predict(faces_roi)
    print(f'label={label} with confidence ={confidence}')

    cv.putText(img, str(people[label]),(x,y),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),3)
    #print(x,y,w,h)

    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

cv.imshow('final',img)

cv.waitKey(0)