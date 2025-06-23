import math
import cv2 as cv
import numpy as np

class EdgeExtractor(object):
    """Detects edge points of an image, by using a Canny edge detector, and returns them as a list of unorganized points"""

    def __init__(self, canny1, canny2=None):
        self.__roi = None
        self.__canny1 = canny1
        if canny2==None:
            self.__canny2 = canny1*3
        else:
            self.__canny2 = canny2;

#################################################################################

    def Extract(self, gray:np.array) -> list((float, float)):
        """
        INPUT: 
        - gray: grayscale image
        - roi: tuple of 2 points, in turn defined as a tuple (x,y), (top-left, bottom-right); if roi is empty, edges are computed over the whole image
        OUTPUT:
        - list of edge points, each of which is defined as a tuple (x,y)
        """

        ymax, xmax = gray.shape;
        ymax -= 1; xmax -=1;
        if self.__roi == None:
            xmin=0; ymin=0;
        else:
            topLeft, bottomRight = self.__roi
            xmin, ymin = map(int, max((0,0), min((xmax, ymax),topLeft)))
            xmax, ymax = map(int, max((0,0), min((xmax, ymax), bottomRight)))

        cropped = gray[ymin:ymax, xmin:xmax]
        imgc = cv.Canny(cropped, self.__canny1, self.__canny2) 
        rowsc, colsc = cropped.shape

        x = []; y=[]
        for r in range(ymax-ymin):
            for c in range(xmax-xmin):
                if r<rowsc and c<colsc and imgc[r,c]>0:
                #if imgc[r,c]>0:
                    x.append(c+xmin)
                    y.append(r+ymin)

        return list(zip(x,y))

#################################################################################

    def SetRoi(self, C: np.array):
        from LevelEstimator import LevelEstimator

        if np.array_equal(C, LevelEstimator.EmptyConicMatrix()):
            self.__roi = None
        else:
            self.__roi = LevelEstimator.GetRoi(C)

    @property
    def CannyThreshold(self):
        return self.__canny1;
    @CannyThreshold.setter
    def CannyThreshold(self, canny: float):
        self.__canny1 = canny
        self.__canny2 = canny/2


