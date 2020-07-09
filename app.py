import cv2
import matplotlib.pyplot as plt
import sys
import dlib
import numpy as np


image = sys.argv[1]
cascades = 'haarcascade_frontalface_default.xml'
smilec = 'haarcascade_smile.xml'
faceCascade = cv2.CascadeClassifier(cascades)
smile_cascade = cv2.CascadeClassifier(smilec)

predictor = "shape_predictor_68_face_landmarks.dat"
pdr = dlib.shape_predictor(predictor)

print(pdr)

img = cv2.imread(image)
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

facedata = faceCascade.detectMultiScale(
        grayimg,
        scaleFactor = 1.2,
        minNeighbors = 7,
        minSize = (20,20),
        flags =  cv2.CASCADE_SCALE_IMAGE
        )

print('found ',len(facedata),'faces!')
print(facedata)
landmark = img.copy()
for ct,(x, y, w, h) in enumerate(facedata):
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img,'Face'+str(ct),(x-3,y-3),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,255,0) )
    dlib_rect = dlib.rectangle(int(x),int(y),int(x+w), int(y+h))
    #print(dlib_rect)

    d_landmarks = pdr(img,dlib_rect).parts()
    #print(d_landmarks)

    landmarks = np.matrix([[p.x,p.y] for p in d_landmarks])
    
    
    print('ct',ct)
    #print(landmarks)
   
    for idx, point in enumerate(landmarks):
            pos = (point[0,0], point[0,1] )
            #annotate the positions
            if idx == 30:
                omg = 'nose/nose4.png'
                oimg = cv2.imread(omg)
                x_offset=y_offset=30
                print('x',x_offset,'y',y_offset,'l.s.0',landmark.shape[0],'l.s.1',landmark.shape[1])
                #landmark[y_offset:y_offset+landmark.shape[0], x_offset:x_offset+landmark.shape[1]] = oimg
               # cv2.addWeighted('nose/nose4.png',0.5,'nose/nose4',0.5,0)
                
            ret = cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
            #print(ret,end='')
    
    print()

#detect smiling 

for(x,y,w,h) in facedata:
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255, 0, 0),2)
    roi_gray = grayimg[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    smile_frames = smile_cascade.detectMultiScale(roi_gray,1.8,20)

    for(sx,sy,sw,sh) in smile_frames:
        cv2.rectangle(roi_color,(sx,sy),((sx+sw),(sy+sh)),(0, 0, 255),2)
    cv2.imwrite('smileimg.jpg',img)


#cv2.waitKey(0)

cv2.imwrite('output/face_detected.png', img)
cv2.imwrite('output/points.png',landmark)
cv2.imwrite('output/grayimage.png',grayimg)

print()

#cv2.destroyAllWindows()

