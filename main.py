import cv2

face_cascade = cv2.CascadeClassifier("data/data.xml")
cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    marks = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in marks:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('image',img)
    cv2.waitKey(0)

cap.release()