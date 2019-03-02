import numpy as np
import cv2
from matplotlib import pyplot as plt

#load images
img1 = cv2.imread('taj.jpg')
img2 = cv2.imread('taj_large.jpg')

#resize
img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25)
img2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)

#orb object
object = cv2.ORB_create()
keyA, destination1 = object.detectAndCompute(img1, None)
keyB, destination2 = object.detectAndCompute(img2, None)
#matches using bruteforce like Harris
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = brute_force.match(destination1, destination2)
matches = sorted(matches, key=lambda x: x.distance)
#choose 20 points to match
similar = cv2.drawMatches(img1, keyA, img2, keyB, matches[:5], outImg=None, flags=2)

#Sift object
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detectAndCompute(img1, None)
kp2 = sift.detectAndCompute(img2, None)

#choose 20 points to match
#siftsimilar = cv2.drawMatches(img1, kp, img2, kp2, matches[:40], outImg=None, flags=2)

cv2.imshow('Harris Match', similar)
#cv2.imshow('SIFT Match', siftsimilar)
c = cv2.waitKey(10000)
cv2.destroyAllWindows()

