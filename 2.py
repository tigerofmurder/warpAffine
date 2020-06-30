import cv2
import numpy as np
import math
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def warp_my(filename,M,rsize):
    img = filename
    rows, cols = rsize
    rows1, cols1 = img.shape[:2]
    img_output = np.zeros((rsize[0],rsize[1],3), np.uint8)
    X1 = np.float32([0,0])
    A = np.float32([ [M[0,0],M[0,1]], [M[1,0],M[1,1] ]])
    
    for x in range(rows-1):
        for y in range(cols-1):
            Y1 = np.float32([ [x-M[0,2]],[y-M[1,2]] ])
            cv2.solve(A,Y1,X1)
            newx = int(X1[0])
            newy = int(X1[1])
            if(newx>0 and newx < rows1 and newy < cols1 and newy>0):
                img_output[x,y,0] = img[newx,newy,0]
                img_output[x,y,1] = img[newx,newy,1]
                img_output[x,y,2] = img[newx,newy,2]
    return img_output
    
def rotation(filename, angle, scale = 1):
    angle = 360-angle
    img = cv2.imread(filename)
    rows, cols = img.shape[:2]

    alpha = scale*math.cos(math.radians(angle))
    betha = scale*math.sin(math.radians(angle))
    center_x = cols/2
    center_y = rows/2
    M = np.ndarray(shape=(2,3), buffer = np.array([[alpha,betha,(1-alpha)*center_x-betha*center_y],[-betha,alpha,betha*center_x+(1-alpha)*center_y]]))
    
    
    newX,newY = rows*scale,cols*scale
    sen = math.sin(math.radians(angle))
    cos = math.cos(math.radians(angle))
    newX,newY = (abs(sen*newY) + abs(cos*newX),abs(sen*newX) + abs(cos*newY))
    
    (tx,ty) = ((newX-rows)/2,(newY-cols)/2)
    M[0,2] += tx
    M[1,2] += ty
    
    img_output = warp_my(img, M, (int(newX),int(newY)))
    strr = "rotation_"+str(random.random())+".jpg"
    cv2.imwrite(strr,img_output)
    return strr


filename = askopenfilename()

rotation(filename,30)
