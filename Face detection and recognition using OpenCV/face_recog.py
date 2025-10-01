import cv2 as cv
import numpy as np
import os

people=['ben','madona']
DIR=r'C:\Users\kashi\Downloads\OPENCV\faces\train'
haar_cascade=cv.CascadeClassifier('haar_face.xml')

features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label= people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)

            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            #cv.imshow('gray',gray)

            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            #print(len(faces_rect))

            for (x,y,w,h) in faces_rect:
               cv.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),1)

               faces_roi=gray[y:y+h,x:x+w]
               features.append(faces_roi)
               labels.append(label)

create_train()
print('TRAINING DOOONNNEEEEEE')

features=np.array(features,dtype='object')
labels=np.array(labels)

face_recog=cv.face.LBPHFaceRecognizer_create()

face_recog.train(features,labels)

face_recog.save('face_trained.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)

cv.waitKey(0)