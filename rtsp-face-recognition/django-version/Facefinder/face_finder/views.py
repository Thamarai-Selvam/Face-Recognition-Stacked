from django.shortcuts import render

# Create your views here.
import numpy as np
import face_finder.face_rec as face_rec
from cv2 import *
import urllib.request as u

import json
import os
from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt




def read_image(path=None,stream=None,url=None):
    if path is not None:
        image = cv2.read(path)
    else:
        if url is not None:
            response = u.urlopen(url)

            data_temp = response.read()

        elif stream is not None:

            data_temp = stream.read()

        image = np.asarray(bytearray(data_temp),dtype="uint8")

        image = cv2.imdecode(image,cv2.IMREAD_COLOR)
    
    return image



@csrf_exempt
def requested_url(request):

    default = {"executed_successfully": False}
    read_url = ""
    image_to_read = ""
    if request.method == "POST":


        read_url = request.GET('text_box')
        if request.FILES.get("image",None) is not None:

            image_to_read = read_image(stream = request.FILES["image"])

        else:
            url_provided = request.POST.get("url",None)

            if url_provided is None:
                default["error_value"] = "There is no URL Provided"

                return JsonResponse(default)

            image_to_read = read_image(url = url_provided)


    url = read_url

    video = cv2.VideoCapture(url)


    #loading sample pictures
    f1 = face_rec.load_image_file(image_to_read)


    #learn how to recognise it
    f1_encoding =  face_rec.face_encodings(f1)[0]

    #array for known encodings
    kfe = []

    #array for known face names
    kfn = []



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

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)# Draw a box around the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (136, 227, 182), 1) #label the face

        

        h,w,l = frame.shape
        new_h = int(h/2)
        new_w = int(w/2)
        rzframe = cv2.resize(frame,(new_w,new_h))
        cv2.imshow('Video', frame)
        count += 1
        
        default.update({"#of_faces": len(floc),
                       "faces":floc,
                       "executed_successfully":True})
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print('processed ',count,'frames')
        #     video.release()
        #     cv2.destroyAllWindows()
        #     break

    return JsonResponse(default)


