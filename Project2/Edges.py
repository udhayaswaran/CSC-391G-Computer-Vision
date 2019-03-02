import cv2
import numpy as np
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam") #If not, throw an error
while True: #While the webcam is open...
    ret, frame = cap.read() #Video capture  = ret, frame
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
    gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
    gray_vid = np.float32(gray_vid) #format to np.float32
    cv2.imshow('Input', gray_vid)
    edged_frame = cv2.Canny(frame, 100,200)
    corner_frame = cv2.cornerHarris(gray_vid, 3, 3, 0.04)
    frame[corner_frame > 0.01*corner_frame.max()] = [0, 0, 255]
   # gftd = cv2.goodFeaturesToTrack(gray_vid, 25, 0.01, 10)
    #cr = cv2.cornerHarris(board, 2, 3, 1)
    cv2.imshow('Edges', edged_frame)
    #cv2.imshow('Corners', corner_frame - corner_frame.min() / (corner_frame.max() - corner_frame.min()))
    cv2.imshow('Corners', frame)
    #cv2.imshow('GFTD', gftd)

    c = cv2.waitKey(1)
    if c == 27: #ESCAPE KEY
        break

cap.release()
cv2.destroyAllWindows()