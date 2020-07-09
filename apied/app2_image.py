from flask import Flask,request,render_template,send_file
from flask_restful import Resource,Api
#----------------------------#
#Actual app imports#
import numpy as np
import cv2
import sys


#Apply overlay(glass and cigar) image

def Applyoverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src
 
#read image from cli and predict facial landmarks

def remake(ifile):
    print(ifile)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    glass_img = cv2.imread('glass3.png', -1)
    cigar_img = cv2.imread('cigar3.png',-1)
    img = cv2.imread(ifile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, 1.3, 5, 0, (30, 30))
    
    print('found',len(faces),'faces')
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
                glass_min = int(y + 1.5 * h / 5)
                glass_max = int(y + 2.5 * h / 5)
                sh_glass = glass_max - glass_min
     
                cigar_min = int(y + 4 * h / 6)
                cigar_max = int(y + 5.5 * h / 6)
                sh_cigar = cigar_max - cigar_min
     
                fglass = img[glass_min:glass_max, x:x+w]
                fcigar = img[cigar_min:cigar_max, x:x+w]
     
                specs = cv2.resize(glass_img, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
                cigar= cv2.resize(cigar_img, (w, sh_cigar),interpolation=cv2.INTER_CUBIC)
                Applyoverlay(fglass,specs)
                Applyoverlay(fcigar,cigar,(int(w/2),int(sh_cigar/2)))
        filename = 'static/images/image_re.png'
        cv2.imwrite(filename,img)
    print('after for loop')       
    return filename



app = Flask(__name__)


@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/process', methods = ['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
      f = request.form['ifile']
      gfile = remake(f)
    #return gfile
    return render_template('index.html',sfile=gfile)


if __name__ == '__main__':
    app.debug = True
    app.run()


