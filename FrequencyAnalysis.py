"""
UDHAYASWARAN PONSHUNMUGAM
COMPUTER VISION
DR PAUCA
LAB 1 PART 2
FREQUENCY ANALYSIS
"""
import cv2
import numpy as np
import numpy
import matplotlib.pyplot as plt
import SpatialFilter
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread('DSC_9259-0.02.JPG',cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = gray_img
grayNoisy = np.zeros(gray_img.shape, np.float64)


fourierint = np.fft.fft2(img)
fshift = np.fft.fftshift(fourierint)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('INPUT'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('MAGNITUDE'), plt.xticks([]), plt.yticks([])
plt.show()

def getdft(gray_img):
    DFT2D = numpy.fft.fft2(gray_img.astype(float))
    return DFT2D
def plot(coefficients,gray_img):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Y = (numpy.linspace(-int(gray_img.shape[0] / 2), int(gray_img.shape[0] / 2) - 1, gray_img.shape[0]))
    X = (numpy.linspace(-int(gray_img.shape[1] / 2), int(gray_img.shape[1] / 2) - 1, gray_img.shape[1]))
    X, Y = numpy.meshgrid(X, Y)
    ax.plot_surface(X, Y, numpy.fft.fftshift(numpy.abs(coefficients)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_surface(X, Y, numpy.fft.fftshift(numpy.log(numpy.abs(coefficients) + 1)), cmap=plt.cm.coolwarm, linewidth=0,
                    antialiased=False)
    plt.show()
    plt.savefig(plt, 'PupFT.JPG')

def plotlogmagnitude(coefficients,mName,logplotname):
    mImage = numpy.fft.fftshift(numpy.abs(coefficients))
    mImage = mImage / mImage.max()
    mImage = ski.img_as_ubyte(mImage)
    cv2.imshow('mag', mImage)
    logmImage = numpy.fft.fftshift(numpy.log(numpy.abs(coefficients) + 1))
    logmImage = logmImage / logmImage.max()
    logmImage = ski.img_as_ubyte(logmImage)
    cv2.imshow('log mag', logmImage)
    cv2.waitKey(0)
    SpatialFilter.saveimg(mName,mImage)
    SpatialFilter.saveimg(logplotname,logmImage)

coefficients = getdft(grayNoisy)
coefficients = fourierint
plot(coefficients, gray_img)
plotlogmagnitude(coefficients,'a.JPG','b.JPG')
