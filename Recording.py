import numpy as np
import cv2
import time
from myfun import camThread,camPreview

# Cam No
camnum=1
# Create an object to read
# from camera
video = cv2.VideoCapture(camnum)#, cv2.CAP_DSHOW) # this is the magic!
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure on at 0.75
video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Auto exposure off at 0.25
time.sleep(2)
video.set(cv2.CAP_PROP_EXPOSURE, -8)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #1920x1080, 1280x720
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(video.get(cv2.CAP_PROP_EXPOSURE))
r, frame = video.read()
...


# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width,frame_height) #Frame size
fps=30 # Frames per second

print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]) + ', fps:' + str(fps))
video.release()

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
out1 = cv2.VideoWriter('output1.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, size)
out2 = cv2.VideoWriter('output2.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, size)
thread1 = camThread("Camera 1", camnum, out1, fps, size)
#thread2 = camThread("Camera 2", 2, out2, fps, size)
thread1.start()
#thread2.start()

time.sleep(10)
import rcb4motionplay