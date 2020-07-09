import urllib.request as u
import cv2
import numpy as np
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

url = 'rtsp://192.168.0.102:8080/video'
print(url)

video = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
while True:
    frame =  video.read()
    cv2.imshow('videofeed',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
