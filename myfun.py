import cv2
import threading
import time
import numpy as np
from scipy.optimize import minimize
import os

# Record
class camThread(threading.Thread):
    def __init__(self, previewName, camID, saveVid, fps, size):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.saveVid = saveVid
        self.fps = fps
        self.size = size
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID, self.saveVid, self.fps, self.size)

def camPreview(previewName, camID, saveVid, fps, size):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)#    , cv2.CAP_DSHOW)  # this is the magic!
    #cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
    # Set resolution to 720p
    #cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Auto exposure off at 0.25 and on at 0.75
    #cam.set(cv2.CAP_PROP_EXPOSURE, -5)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    # used to record the time when we processed last frame
    prev_frame_time = time.time()
    imgs=[]
    # Naming a window
    cv2.namedWindow(previewName, cv2.WINDOW_NORMAL)
    # Using resizeWindow()
    cv2.resizeWindow(previewName, 1920, 1080)

    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
         rval = False
    
    flag=0
    # Images from video
    while rval:
        # time when we finish processing for this frame
        new_frame_time = time.time()  # output the frame
        # used to record the time at which we processed current frame
        fps_current = 1 / (new_frame_time - prev_frame_time)
        if (fps_current<=fps):
            cv2.imshow(previewName, frame)
            rval, frame = cam.read()
            print(fps_current)
            #saveVid.write(frame)
            key2 = cv2.waitKey(5)
            # Record on R key
            if key2==114 or flag:
                flag=1
                imgs.append(frame)

            prev_frame_time = new_frame_time
            key = cv2.waitKey(5)
            imgs.append(frame)
            prev_frame_time = new_frame_time
            key = cv2.waitKey(5)
            if key == 27:  # exit on ESC
                i=0
                for img in imgs:
                    new_frame_time = time.time()  # output the frame
                    #print(img.shape)
                    saveVid.write(img)
                    prev_frame_time = time.time()
                    t_overall=(1/fps)-(prev_frame_time-new_frame_time)
                    if t_overall>0:
                        time.sleep(t_overall)
                break
    cv2.destroyWindow(previewName)

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#ut = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Create two threads as follows
#thread1 = camThread("Camera 1", 0,out)
#thread2 = camThread("Camera 2", 1,out)
#thread1.start()
#hread2.start()


# calibration
import glob
# termination criteria
# Intrinsic Calibration using chessboard images
def incalib(images,filename):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    np.multiply(24,objp)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    #images = glob.glob('*.jpeg')
    for img in images:
        #img = cv.imread(fname)
        #cv.imshow('img', img)
        #cv.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    img = images[len(images)-1]#cv.imread('calib.jpeg')
    #h,  w = img.shape[:2]
    #newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)
    cv2.imshow('Default', img)
    cv2.imshow('calibresult', dst)
    cv2.waitKey(100)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )

    # Save Camera Matrix
    print(mtx)
    print(len(tvecs))
    #print(rvecs)
    np.savez(filename, v1=mtx,v2=dist,v3=tvecs,v4=rvecs)
"""cap=cv2.VideoCapture('sample2.avi')#glob.glob('*.jpg')
imgs=[]
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        imgs.append(frame)
    # Break the loop
    else:
        break

#print(sample)
mycalib(imgs,'camparam2.npz')
"""

class camcal:
  def __init__(self, cmtx,dist,tvec,rvec):
    self.cmtx = cmtx
    self.dist = dist
    self.tvec = tvec
    self.rvec = rvec

#Extrinsic Calibration using chessboard
def excalibc(img,objp,cam):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        imgp = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), imgp, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    #return imgp
    success1, cam.rvec, cam.tvec = cv2.solvePnP(objp, imgp.astype('float32'), cam.cmtx, cam.dist, flags=0)
    cam.rmtrx = cv2.Rodrigues(cam.rvec)[0]
    # Transformation from world to cam and image
    E = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    cam.T = np.zeros([4, 4], dtype=float) # From world frame to img frame
    # print(np.linalg.det(cam1.rmtrx))
    cam.T[0:3, 0:3] = np.asarray(cam.rmtrx, dtype=float)
    cam.T[0:3, 3] = np.asarray(cam.tvec, dtype=float).ravel()
    cam.Timg = np.matmul(cam.cmtx, np.matmul(E, cam.T)) # From world frame to pixels
    # print(cam1.rmtrx)
    print('Error_CAM=',np.linalg.norm(imgp - cv2.projectPoints(objp, cam.rvec, cam.tvec, cam.cmtx, cam.dist)[0]))
    img = cv2.drawFrameAxes(img, cam.cmtx, None, cam.rvec, cam.tvec, 100, 3)
    cv2.imshow('Pose', img)
    # cv2.imwrite('test_solvePnP.png', img)
    cv2.waitKey()
    return cam

#Extrinsic Calibration using square corners detected
def excalibs(img,objp,cam):
    imgp = findmark(img)  # excalib(img1,objp)
    #return imgp
    success1, cam.rvec, cam.tvec = cv2.solvePnP(objp, imgp.astype('float32'), cam.cmtx, cam.dist, flags=0)
    cam.rmtrx = cv2.Rodrigues(cam.rvec)[0]
    # Transformation from world to cam and image
    E = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    cam.T = np.zeros([4, 4], dtype=float) # From world frame to img frame
    # print(np.linalg.det(cam1.rmtrx))
    cam.T[0:3, 0:3] = np.asarray(cam.rmtrx, dtype=float)
    cam.T[0:3, 3] = np.asarray(cam.tvec, dtype=float).ravel()
    cam.Timg = np.matmul(cam.cmtx, np.matmul(E, cam.T)) # From world frame to pixels
    # print(cam1.rmtrx)
    print('Error_CAM=',np.linalg.norm(imgp - cv2.projectPoints(objp, cam.rvec, cam.tvec, cam.cmtx, cam.dist)[0]))
    img = cv2.drawFrameAxes(img, cam.cmtx, None, cam.rvec, cam.tvec, 100, 3)
    cv2.imshow('Pose', img)
    # cv2.imwrite('test_solvePnP.png', img)
    cv2.waitKey()

    return cam

# Stereo Calibration % Not working for one set of image..
def stereocalib(objp,img1,img2,cam1,cam2):
    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    width = img1.shape[1]
    height = img1.shape[0]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    c_ret1, corners1 = cv2.findChessboardCorners(gray1, (9, 6), None)
    c_ret2, corners2 = cv2.findChessboardCorners(gray2, (9, 6), None)

    if c_ret1 == True and c_ret2 == True:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        #corners1 = np.array([[corner for [corner] in corners1]])
        corners = corners1.reshape(corners1.shape[0], corners1.shape[2])
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        #corners2 = np.array([[corner for [corner] in corners2]])
        corners2 = corners2.reshape(corners2.shape[0], corners2.shape[2])

        cv2.drawChessboardCorners(img1, (9, 6), corners1, c_ret1)
        cv2.imshow('img', img1)

        cv2.drawChessboardCorners(img2, (9, 6), corners2, c_ret2)
        cv2.imshow('img2', img2)
        k = cv2.waitKey(0)
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objp.astype('float32'), corners1.astype('float32'), corners2.astype('float32'), cam1.cmtx,cam1.dist,cam2.cmtx, cam2.dist, (width, height), criteria=criteria,flags=stereocalibration_flags)

    return R,T

def drawcube(img,cam,fc,face1,face2,face3):
    imgpts=cv2.projectPoints(face1.astype('float32'),cam.rvec,cam.tvec,cam.cmtx,cam.dist)[0]
    cv2.drawContours(img, imgpts.astype('int'),-1, (0, 0, 0), 5)
    imgpts=cv2.projectPoints(face2.astype('float32'),cam.rvec,cam.tvec,cam.cmtx,cam.dist)[0]
    cv2.drawContours(img, imgpts.astype('int'),-1, (0, 0, 0), 5)
    imgpts=cv2.projectPoints(face3.astype('float32'),cam.rvec,cam.tvec,cam.cmtx,cam.dist)[0]
    cv2.drawContours(img, imgpts.astype('int'),-1, (0, 0, 0), 5)
    # Draw Coordinate system
    imgcnt1 = cv2.projectPoints(fc.astype('float32'), cam.rvec, cam.tvec, cam.cmtx, cam.dist)[0].ravel()
    xaxis =  cv2.projectPoints((fc + np.array([30, 0, 0])).astype('float32'), cam.rvec, cam.tvec, cam.cmtx, cam.dist)[0].ravel()
    yaxis = cv2.projectPoints((fc + np.array([0, 30, 0])).astype('float32'), cam.rvec, cam.tvec, cam.cmtx, cam.dist)[0].ravel()
    zaxis = cv2.projectPoints((fc + np.array([0, 0, 30])).astype('float32'), cam.rvec, cam.tvec, cam.cmtx, cam.dist)[0].ravel()
    cv2.line(img, (int(imgcnt1[0]), int(imgcnt1[1])), (int(xaxis[0]), int(xaxis[1])), (0, 0, 255), 5)
    cv2.putText(img, 'X', (int(xaxis[0]), int(xaxis[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.line(img, (int(imgcnt1[0]), int(imgcnt1[1])), (int(yaxis[0]), int(yaxis[1])), (0, 255, 0), 5)
    cv2.putText(img, 'Y', (int(yaxis[0]), int(yaxis[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.line(img, (int(imgcnt1[0]), int(imgcnt1[1])), (int(zaxis[0]), int(zaxis[1])), (255, 0, 0), 5)
    cv2.putText(img, 'Z', (int(zaxis[0]), int(zaxis[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imshow('Image cube Cordinates', img)
    cv2.waitKey(0)

# Detect shape
#from scipy.spatial import distance as dist

from matplotlib import pyplot as plt
# reading image
def order_points(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros(pts.shape, dtype=pts.dtype)
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=2)
	rect[0,:,:] = pts[np.argmin(s,axis=0),:,:]
	rect[2,:,:] = pts[np.argmax(s,axis=0),:,:]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=2)
	rect[1,:,:] = pts[np.argmin(diff,axis=0),:,:]
	rect[3,:,:] = pts[np.argmax(diff,axis=0),:,:]
	# return the ordered coordinates
	return rect
def findmark(img):
    myshape = []
    #img = cv2.imread('cal1.jpg')
    #img = cv2.resize(img, (1280, 720))
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur = cv2.blur(gray, (3, 3)) # blur the image
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    #blur=cv2.GaussianBlur(threshold,(15,15),0)
    #edged = cv2.Canny(threshold, 30, 200)
    #cv2.imshow('im',threshold)
    #cv2.waitKey(0)
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0

    # list for storing names of shapes
    for contour in contours:

        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)


        # finding center point of shape
        #approx = approx.astype('float32')
        #M = cv2.moments(approx)
        #print(approx)
        x=np.mean(approx[:,:,0])
        y=np.mean(approx[:,:,1])
        #if M['m00'] != 0.0:
        #    x = int(M['m10'] / M['m00'])
        #    y = int(M['m01'] / M['m00'])

        # putting shape name at center of each shape
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = float(w) / h
            if (ratio >= 0.8)*(ratio <= 1.2) and np.linalg.norm(([x,y]-approx))>20:
                #print([x, y])
                myshape=order_points(approx)
                #print(approx)
                #print(myshape)
                # using drawContours() function
                cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
                cv2.putText(img, 'Square', (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # Show pts
    circolor = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0]]
    k = 0
    for j in myshape:
        # print(j)
        cv2.circle(img, (j[0, 0], j[0, 1]), 10, circolor[k], 2)
        k = k + 1
    # Naming a window
    cv2.namedWindow("shapes", cv2.WINDOW_NORMAL)
    # Using resizeWindow()
    cv2.resizeWindow("shapes", 1920, 1080)

    # displaying the image after drawing contours
    cv2.imshow('shapes', img)

    cv2.waitKey(50)
    cv2.destroyAllWindows()
    return myshape

#img = cv2.imread('Picture1.jpg')
#result=findmark(img)
#print(result)

def findredcircle(img,num):
    myshape = []
    # Red color
    # FV
    #low_red = np.array([0, 0, 140])
    #high_red = np.array([150, 150, 255])
    # SV
    low_red = np.array([0, 0, 150])
    high_red = np.array([130, 130, 255])
    red_mask = cv2.inRange(img, low_red, high_red)
    red = cv2.bitwise_and(img, img, mask=red_mask)
    red = cv2.medianBlur(red, 3)

    redbw=cv2.inRange(red,low_red,high_red)
    # First blur to reduce noise prior to color space conversion
    redbw = cv2.medianBlur(redbw, 3)

    #img = cv2.imread('cal1.jpg')
    #img = cv2.resize(img, (1280, 720))
    cv2.imshow('shapes', redbw)
    #cv2.waitKey(0)

    detected_circles = cv2.HoughCircles(redbw, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=50, param2=10, minRadius=5, maxRadius=100)

    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        if len(detected_circles[0,:])==num  and np.std(detected_circles[0,:,2])<=1:
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                #cv2.imshow("Detected Circle", img)
                #cv2.waitKey(0)
                myshape.append(pt)
                # putting shape name at center of each shape
                cv2.putText(img, 'Circle', (int(a), int(b)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    #contour=(a,b)
    #print(myshape)
    cv2.namedWindow("shapes", cv2.WINDOW_NORMAL)
    # Using resizeWindow()
    cv2.resizeWindow("shapes", 1920, 1080)
    # displaying the image after drawing contours
    cv2.imshow('shapes', img)
    cv2.waitKey(50)
    cv2.destroyAllWindows()
    return myshape

def findbluecircle(img,num):
    myshape = []
    # blue color
    # FV
    #low = np.array([140, 0, 0])
    #high = np.array([255, 100, 100])
    # SV
    low = np.array([130, 0, 0])
    high = np.array([255, 120, 100])
    col_mask = cv2.inRange(img, low, high)
    col = cv2.bitwise_and(img, img, mask=col_mask)

    bw=cv2.inRange(col,low,high)
    # First blur to reduce noise prior to color space conversion
    bw = cv2.medianBlur(bw, 3)

    #img = cv2.imread('cal1.jpg')
    #img = cv2.resize(img, (1280, 720))
    cv2.imshow('shapes',bw)
    #cv2.waitKey(0)

    detected_circles = cv2.HoughCircles(bw, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=50, param2=10, minRadius=2, maxRadius=50)

    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        if len(detected_circles[0,:])==num and np.std(detected_circles[0,:,2])<=1:
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                #cv2.imshow("Detected Circle", img)
                #cv2.waitKey(0)
                myshape.append(pt)
                # putting shape name at center of each shape
                cv2.putText(img, 'Circle', (int(a), int(b)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    #contour=(a,b)
    #print(myshape)
    cv2.namedWindow("shapes", cv2.WINDOW_NORMAL)
    # Using resizeWindow()
    cv2.resizeWindow("shapes", 1920, 1080)
    # displaying the image after drawing contours
    cv2.imshow('shapes', img)
    cv2.waitKey(50)
    cv2.destroyAllWindows()
    return myshape
def vid2img(vid,crop,saveimgs=0):
    if (vid.isOpened() == False):
        print("Error opening video stream or file")
    i=0
    imgs=[]
    # frame
    currentframe = 0
    if saveimgs==True:
        try:
            # creating a folder named data
            if not os.path.exists('vid2img'):
                os.makedirs('vid2img')

        # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')
    # Naming a window
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # Using resizeWindow()
    cv2.resizeWindow("Frame", 1920, 1080)
    # Read until video is completed
    while (vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()
        if ret == True:
            imgs.append(frame)
            #findmark(frame)
            i=i+1

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1)
            #if key == 27:  # exit on ESC
            #    break
            # Save the frame if savevid=1
            if saveimgs==1:
                # if video is still left continue creating images
                name = './vid2img/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)

                # writing the extracted images

                cv2.imwrite(name, frame[crop[0]:crop[1],crop[2]:crop[3]])
                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1


        # Break the loop
        else:
            break
    #print(i)
    # When everything done, release the video capture object
    #vid.release()
    return imgs

# Find world coordinates from image
def img2weqn(u,v,Timg):
    # Ax=b -- A [x,y,z] = p [u,v,1] -- A [sx,sy,sz] = [u v 1]
    o_sw=np.matmul(np.linalg.pinv(Timg),np.append([u,v],1)) # world coord of random pt on line
    o_sw=o_sw/o_sw[3]
    return o_sw[0:3]


def myobj(scal,os1,tcam1,os2,tcam2,lcube):
    oc1=tcam1+scal[0]*(os1-tcam1)+[-lcube,0,0]
    oc2=tcam2+scal[1]*(os2-tcam2)+[0,-lcube,0]
    # Shortest distance bw these two vectors
    #D=np.zeros([3,3])
    #D[0,:]=[]
    d=np.linalg.norm(oc1-oc2) #np.linalg.det(D)/np.sqrt(np.linalg.det(D[1:3,0:2])**2+np.linalg.det(D[1:3,[0,2]])**2+np.linalg.det(D[1:3,1:3])**2)
    return d

def triangl(cam1,cam2,imgpts1,imgpts2,lcube):
    # Center of squares in world frame
    os1 = img2weqn(int(np.mean(imgpts1[:, :, 0])), int(np.mean(imgpts1[:, :, 1])), cam1.Timg)
    os2 = img2weqn(int(np.mean(imgpts2[:, :, 0])), int(np.mean(imgpts2[:, :, 1])), cam2.Timg)
    # Cam  in world frame
    tcam1=-np.matmul(np.transpose(cam1.rmtrx),cam1.tvec).ravel()
    tcam2=-np.matmul(np.transpose(cam2.rmtrx),cam2.tvec).ravel()
    # Solve minimization problem
    res = minimize(myobj, [0,0], args=(os1.ravel(),tcam1,os2.ravel(),tcam2,lcube))
    print('Min distance is (error):',res.fun)
    oc1=tcam1 + res.x[0] * (os1.ravel() - tcam1) + [-lcube, 0, 0]
    oc2=tcam2 + res.x[1] * (os2.ravel() - tcam2) + [0, -lcube, 0]
    oc_track = np.mean([oc1,oc2],axis=0)
    print(oc_track)
    return oc_track

