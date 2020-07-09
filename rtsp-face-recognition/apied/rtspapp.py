from flask import Flask,request,render_template,send_file
from flask_restful import Resource,Api


import numpy as np
import face_rec
import cv2
import urllib.request as u



def findout(image,iname):

    print(image)
    url = 'rtsp://192.168.43.1:8080/h264_ulaw.sdp'
    # url = 'http://192.168.43.1:8080/video'
    video = cv2.VideoCapture(url)


    #loading sample pictures
    f1 = face_rec.load_image_file(image)


    #learn how to recognise it
    f1_encoding =  face_rec.face_encodings(f1)[0]

    #array for known encodings
    kfe = [f1_encoding]

    #array for known face names
    kfn = [iname]



    names = []
    flag = True
    floc = []
    fe = []
    count = 0
    while(1):

        ret,frame = video.read() #grab frame by frame while(1)

        rframe = cv2.resize(frame,(0,0),fx=0.25,fy=0.25) #not needed ,
                                                        #just to make the process faster 

        rgbrframe = cv2.cvtColor(rframe,cv2.COLOR_BGR2RGB)#cv2 uses BGR color whereas,
                                    #face_rec uses RGB , so reverse content

        if flag:
            floc = face_rec.face_locations(rgbrframe) # grab face from frame
            fe   = face_rec.face_encodings(rgbrframe,floc) # grab face encodings from frame 
            
            names = []
            for fenc in fe:
                matched_faces = face_rec.compare_faces(kfe,fenc)
                name = 'Unknown'

                fdist = face_rec.face_distance(kfe,fenc)
                best_match = np.argmin(fdist)
                if matched_faces[best_match]:
                    name = kfn[best_match]

                names.append(name)
        flag = not flag

        # Display the results
        for (top, right, bottom, left), name in zip(floc, names):
            top *= 4    # resize image back again by *0.25
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)# Draw a box around the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) #label the face
            
        

        h,w,l = frame.shape
        new_h = int(h/2)
        new_w = int(w/2)
        rzframe = cv2.resize(frame,(new_w,new_h))
        cv2.imshow('Cam_feed', rzframe)
        count += 1
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('processed ',count,'frames')
            video.release()
            cv2.destroyAllWindows()
            break




app = Flask(__name__)

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/process', methods = ['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
      f = request.form['ifile']
      n = request.form['name']
      findout(f,n)
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
