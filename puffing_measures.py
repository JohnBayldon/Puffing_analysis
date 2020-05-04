from PIL import Image
import os,sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

IMAGE_DIR = "Z:/Materials/Current Projects/cracking/Microscopy/puffing"
IMAGE_FILE = "10#1_b_a.jpg"


def image_measure(file_name = None):
    if file_name is None:
        file_name= os.path.join(IMAGE_DIR,IMAGE_FILE)
    if not os.path.isfile(file_name):
        return -1
    myIm = cv2.imread(file_name)
    gray = cv2.cvtColor(myIm,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200,apertureSize=3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200,min_theta= np.pi/2-0.1,max_theta=np.pi/2+0.1)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        print(f"x0={x0},y0={y0},theta={theta}")
        x1 = 0
        y1 = int(y0+x0*a*a/b-x1*a/b)
        x2 = 1600
        y2 = int(y0+x0*a*a/b-x2*a/b)

        cv2.line(myIm, (x1, y1), (x2, y2), (0, 0, 255), 2)
    print(f"{x1},{y1},{x2},{y2}")
    y_min=min(y1,y2)
    print(f"{x1},{y1-y_min+20},{x2},{y2-y_min+20}")
    return (myIm[(y_min-20):(y_min+150),:],gray[(y_min-20):(y_min+150),:],[x0,y0-y_min+20,theta])

def find_top(im,threshold = 1500,x0=0,y0=0,theta=0):
    """
    finds the top of the part
    :param im:
    :param threshold:
    :return:
    """
    sobely = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=5)
    local_maxes = argrelextrema(sobely,axis=0,comparator= np.greater,order=5)
    maxes=[i for i in range(len(local_maxes[0])) if (sobely[local_maxes[0][i],local_maxes[1][i]])>threshold]
    best_line = np.zeros([1600,3],dtype=np.int32)
    a=np.cos(theta)
    b=np.sin(theta)
    for max in maxes:
        curr_x = local_maxes[0][max]
        curr_y = local_maxes[1][max]
        if best_line[curr_y,0] == 0:
            best_line[curr_y,0] = curr_x
            best_line[curr_y,1] = curr_y
            best_line[curr_y,2] = (curr_y-x0)*a+(curr_x-y0)*b
        elif curr_x>best_line[curr_y,0]:
            best_line[curr_y,0] = curr_x
            best_line[curr_y,1] = curr_y
            best_line[curr_y, 2] = (curr_y - x0) * a + (curr_x - y0) * b
        if curr_y==126:
            print(best_line[curr_y,:],x0,y0,a,b)

    print('testing')
    #return [local_maxes[0][maxes],local_maxes[1][maxes]]
    return  best_line

if __name__ == "__main__" :
    file_list = os.listdir(IMAGE_DIR)
    for f_n in file_list[16:20]:

        file_name = os.path.join(IMAGE_DIR,f_n )
        (im1,im2,line_data) = image_measure(file_name)
        max_pos = find_top(im2,x0=line_data[0],y0=line_data[1],theta= line_data[2])
        sobely = np.flip(cv2.Sobel(im2,cv2.CV_64F,0,1,ksize=5),axis=0)
        f,ax = plt.subplots(3,1)
        f.suptitle(f'{f_n[:-4]}')
        f.canvas.set_window_title(f"{f_n[:-4]}")
        im1[max_pos[:,0],max_pos[:,1]]=[255,0,0]
        ax[0].imshow(np.flip(im1,axis=0))
        ax[1].imshow(sobely, cmap='gray')
        ax[2].plot(max_pos[:,1],max_pos[:,2],'r.',markersize = 1)
        local_maxes = argrelextrema(sobely[:,600],np.greater,order=5)[0]
        #print(max_pos[:,2])
        with open(f"output_{f_n[:-4]}.txt",'w') as f:
            for i in range(max_pos.shape[0]):
                f.write(f"{max_pos[i,0]},{max_pos[i,1]},{max_pos[i,2]}\n")
                #print(max_pos[i,:])
        plt.show()