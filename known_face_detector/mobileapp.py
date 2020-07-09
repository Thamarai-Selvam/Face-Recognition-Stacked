import numpy as np
import face_rec
import cv2
import urllib.request as u



url = 'http://192.168.0.102:8080/shot.jpg'






#loading sample pictures
f1 = face_rec.load_image_file('faces/modi.jpg')
f2 = face_rec.load_image_file('faces/nagesh.jpg')
f3 = face_rec.load_image_file('faces/trump.jpg')
f4 = face_rec.load_image_file('faces/yb.jpg')
f5 = face_rec.load_image_file('faces/kamal.jpg')
f6 = face_rec.load_image_file('faces/swarna.png')
f7 = face_rec.load_image_file('faces/atchaya.png')
f8 = face_rec.load_image_file('faces/js.png')
f9 = face_rec.load_image_file('faces/rethanya.png')
f10 = face_rec.load_image_file('faces/swetha.png')
f11 = face_rec.load_image_file('faces/thamarai.png')
f12 = face_rec.load_image_file('faces/hariharan.png')
f13 = face_rec.load_image_file('faces/priyesh.png')
f14 = face_rec.load_image_file('faces/thilak.png')
f15 = face_rec.load_image_file('faces/veno.png')


#learn how to recognise it
f1_encoding =  face_rec.face_encodings(f1)[0]
f2_encoding =  face_rec.face_encodings(f2)[0]
f3_encoding =  face_rec.face_encodings(f3)[0]
f4_encoding =  face_rec.face_encodings(f4)[0]
f5_encoding =  face_rec.face_encodings(f5)[0]
f6_encoding =  face_rec.face_encodings(f6)[0]
f7_encoding =  face_rec.face_encodings(f7)[0]
f8_encoding =  face_rec.face_encodings(f8)[0]
f9_encoding =  face_rec.face_encodings(f9)[0]
f10_encoding =  face_rec.face_encodings(f10)[0]
f11_encoding =  face_rec.face_encodings(f11)[0]
f12_encoding =  face_rec.face_encodings(f12)[0]
f13_encoding =  face_rec.face_encodings(f13)[0]
f14_encoding =  face_rec.face_encodings(f14)[0]
f15_encoding =  face_rec.face_encodings(f15)[0]

#array for known encodings
kfe = [
    f1_encoding,
    f2_encoding,
    f3_encoding,
    f4_encoding,
    f5_encoding,
    f6_encoding,
    f7_encoding,
    f8_encoding,
    f9_encoding,
    f10_encoding,
    f11_encoding,
    f12_encoding,
    f13_encoding,
    f14_encoding,
    f15_encoding
]

#array for known face names
kfn = [
    'Modi Ji',
    'Nagesh',
    'Trump',
    'yogi babu',
    'Kamal Haasan',
    'Swarna',
    'Atchaya',
    'Jeyasri',
    'Rethanya',
    'Swetha',
    'Thamarai',
    'Hari Haran',
    'Priyesh',
    'Thilak',
    'Veno'
]



names = []
flag = True
floc = []
fe = []

while(1):

#start webcam
    video_url = u.urlopen(url)
    videof = np.array(bytearray(video_url.read()),dtype = np.uint8)
    video = cv2.imdecode(videof,-1)
    frame = video #grab frame by frame while(1)

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

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)# Draw a box around the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (136, 227, 182), 1) #label the face

    
    cv2.imshow('Video', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()