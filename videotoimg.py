import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from myfun import findmark,vid2img
#vid = cv2.VideoCapture('output2.avi')
#vid = cv2.VideoCapture('cam2_SV.avi')
import img2real
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#vid = cv2.VideoCapture('sample.mp4')
#vid = cv2.VideoCapture(0)
# Check if camera opened successfully

# Load Camera parameters for img to world transformation
#campar = np.load('cam2world.npz', boolean='true')
cam1=img2real.cam1
cam2=img2real.cam2
lcube=img2real.lcube

# Load video1
vid1 = cv2.VideoCapture('cam1_FV.avi')
vid2 = cv2.VideoCapture('cam2_SV.avi')

# Load images in video
imgs1 = vid2img(vid1)
imgs2 = vid2img(vid2)

# When everything done, release the video capture object
vid1.release()
vid2.release()

# Cube center for Tracking
oc_track=[]
t=[] #time
k=0
for i in range(110,150):#min(len(imgs1),len(imgs2))):
    img1=imgs1[i]
    j=i-20
    img2=imgs2[j]
    # Find Square shape in picture
    imgpts1 = findmark(img1)
    imgpts2 = findmark(img2)
    if len(imgpts1)>0 and len(imgpts2)>0:
        # Triangulation
        oc_track.append(img2real.triangl(cam1, cam2, imgpts1, imgpts2, lcube))
        t.append(k/10)
    k=k+1

oc_hip=np.asarray(oc_track)
print(t)
print(oc_track)
scipy.io.savemat('hiptraj_exp.mat', dict(t=t, oc_exp=oc_hip))
plt.plot(t,oc_hip[:,0],'ro', label='X')
plt.plot(t,oc_hip[:,1],'go', label='Y')
plt.plot(t,oc_hip[:,2],'bo', label='Z')
plt.xlabel('Time (s)')
plt.ylabel('Distance (mm))')
# Function add a legend
plt.legend()
# function to show the plot
plt.show()# Closes all the frames
#cv2.destroyAllWindows()

