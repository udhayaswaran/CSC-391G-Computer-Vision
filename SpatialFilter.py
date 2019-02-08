"""
LAB 1
PART ONE - SPATIAL FILTER
UDHAYASWARAN PONSHUNMUGAM
"""
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import skimage as ski


#READ IN IMAGE, YOU CAN RESIZE IT
image = cv2.imread("DSC_9259-0.02.jpg") #read in the image at specified place and assign to 'image'; in this case, it is within our project directory
nature_image = cv2.imread("window-00-00.jpg")
puppy_noised = cv2.imread("pnoised.JPG")
small = cv2.resize(image, (0,0), fx=0.25, fy=0.25) #We can resize the image and make it so that the image is only a quarter of the output
cv2.imshow('Original Puppy', image)

k = int(input("Define a size k"))
#2D BLUR, BUILT IN FUNCTION
blur = cv2.blur(image,(k,k))
print("Here is the blurred image")
cv2.imshow("Blurred image", blur)
cv2.imwrite('blurry.JPG', blur)
#cv2.imwrite('Blurry puppy', blur)

key = cv2.waitKey(100)

#2D - AFFINE TRANSFORMATION
rows, cols = image.shape[:2]
source = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
destination = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1],[int(0.66*cols),rows-1]])
projective_matrix = cv2.getPerspectiveTransform(source, destination)
img_output = cv2.warpPerspective(image, projective_matrix, (cols,rows))

cv2.imshow('Affine Transformation', img_output)
cv2.imwrite('AffineTransform.JPG', img_output)

#GAUSSIAN BLUR
gaussian_blur = cv2.GaussianBlur(image, (9, 9), 0)
cv2.imshow('Gaussian blur', gaussian_blur)
cv2.imwrite('gaussian_puppy.JPG', gaussian_blur)

#MEDIAN BLUR
median = cv2.medianBlur(image, 17)
cv2.imshow("median", median)
cv2.imwrite('median_puppy.JPG', median)

#CANNY EDGES
canny_puppy = cv2.Canny(image, 100, 200)
cv2.imwrite('canny_puppy.JPG', canny_puppy)
canny_puppy_noised = cv2.Canny(puppy_noised, 100, 200)
cv2.imwrite('canny_puppy_noised.JPG', canny_puppy_noised)
canny_nature = cv2.Canny(nature_image, 100, 200)
cv2.imwrite('canny_nature.JPG', canny_nature)

cv2.imshow("Canny puppy", canny_puppy)
cv2.imshow("Canny puppy noised", canny_puppy_noised)
cv2.imshow("Canny nature", canny_nature)

key = cv2.waitKey(30000)
cv2.destroyAllWindows()




