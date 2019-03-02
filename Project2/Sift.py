import cv2
import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imread, imresize

#pd.set_option('display.max_rows', 10)
img = cv2.imread('elephant_upscale.png')
if img is None:
    print('Unable to load image.')
    quit()
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
sift = cv2.xfeatures2d.SIFT_create()
# Get the Key Points from the 'gray' image, this returns a numpy array
kp = sift.detect(gray, None)
# Now we drawn the gray image and overlay the Key Points (kp)
img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Plot it to the screen, looks a little small
#plt.imshow(img)
cv2.imwrite('sift.jpg', img)