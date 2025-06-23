import cv2 as cv
import numpy as np
import math

class PyramidOfGaussians(object):
    """Build a Piramid of downsampled and Gaussian-filtered images from an input image"""

    def __init__(self, image, levels):
        rows, cols = image.shape
        maxLevel = 1
        while (rows//2)*2 == rows and (cols//2)*2 == cols:
            maxLevel = maxLevel+1
            rows = rows//2
            cols = cols//2

        self.__levels = max(1,min(maxLevel,levels))
        self.__BuildPyramid(image)

#################################################################################

    @property
    def Levels(self):
        return self.__levels;

#################################################################################

    def __BuildPyramid(self, image):
        """Build the pyramid of downsampled images, made by <levels> levels. 
        The original image is at level 0; the smallest one at level <levels>-1"""

        self.__pyramid = []
        self.__T = []

        self.__pyramid.append(image)
        self.__T.append(np.eye(3))
        for lev in range(1,self.__levels):
            rows, cols = self.__pyramid[lev-1].shape
            down = cv.pyrDown(self.__pyramid[lev-1], dstsize=(cols // 2, rows // 2), borderType=cv.BORDER_REFLECT)
            self.__pyramid.append(down)

            s = max(rows, cols)
            Tlev = np.array([[2/s,   0, -cols/s], 
                             [  0, 2/s, -rows/s], 
                             [  0,   0,       1]])
            self.__T.append(Tlev);

#################################################################################

    def GetLevelImage(self, level):
        """Returns the downsamples image at level <level>"""
        lev = max(0, min(level, self.Levels-1))
        return self.__pyramid[lev]


#################################################################################






