import cv2 as cv

#CAPTURING VIDEO

video=cv.VideoCapture(0)
haar_cascade= cv.CascadeClassifier('haar_face.xml')

while True:
    isTRUE ,frames=video.read()
    cv.imshow('video',frames)

    grey=cv.cvtColor(frames,cv.COLOR_BGR2GRAY)
    cv.imshow('grey',grey)

    faces_rect=haar_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=3)
    print(len(faces_rect))

    for (x,y,w,h) in faces_rect:
        cv.rectangle(grey,(x,y),(x+w,y+h),(0,255,0),1)

    cv.imshow('grey',grey)

    if(cv.waitKey(20) & 0xFF==ord('d')):
        break

video.release()
cv.destroyAllWindows()

#img=cv.imread('PHOTOS/lady.jpg')
#cv.imshow('lady',img)

"""while True:

    isTrue, grey=cv.cvtColor(frames,cv.COLOR_BGR2GRAY)
    cv.imshow('grey',grey)
 
    if(cv.waitKey(20) & 0xFF==ord('d')):
        break"""

"""while True:
    isTRUE ,frames=grey.read()
    cv.imshow('video',frames)

    if(cv.waitKey(20) & 0xFF==ord('d')):
        break"""

#haar_cascade=cv.CascadeClassifier('haar_face.xml')
"""
faces_rect=haar_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=3)
print(len(faces_rect))

for (x,y,w,h) in faces_rect:
    cv.rectangle(grey,(x,y),(x+w,y+h),(0,255,0),1)

cv.imshow('grey',grey)"""
"""while True:
    isTRUE ,frames=grey.read()
    cv.imshow('video',frames)

    if(cv.waitKey(20) & 0xFF==ord('d')):
        break"""

cv.waitKey(20)