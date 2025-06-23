import math
import numpy as np
import cv2 as cv
from enum import Enum, auto

from PyramidOfGaussians import PyramidOfGaussians
from EdgeExtractor import EdgeExtractor
from LevelEstimatorCreator import LevelEstimator, LevelEstimatorCreator, EstimationMode
from Viewer import Viewer

from Settings import CANNY_THRESHOLD

class SphereDetector(object):
    """description of class"""

    def __init__(self, estimMode: EstimationMode):
        
        #print(LevelEstimator._SetOutlierThreshold(1))

        if (estimMode == EstimationMode.HOUGH):
            raise TypeError("Hough transform not allowed (only for internal initialization)")

        canny = CANNY_THRESHOLD
        self.__edge = EdgeExtractor(canny, canny/2)
        self.__boot = LevelEstimatorCreator.Factory(EstimationMode.HOUGH, self.__edge) 
        self.__optim = LevelEstimatorCreator.Factory(estimMode, self.__edge) 
        self.__viewer = Viewer();
        self.__boot.Attach(self.__viewer);
        self.__optim.Attach(self.__viewer);

    @property
    def Viewer(self):
        return self.__viewer

    def Localize(self, image: np.array, levels: int) -> (float, float, float, float, float):

        LevelEstimator.TotalIterations = 0

        pyramid = PyramidOfGaussians(cv.cvtColor(image, cv.COLOR_BGR2GRAY), levels)
        top = pyramid.Levels-1
        self.__boot.Level = top;
        for lev in reversed(range(pyramid.Levels)):

            self.__optim.Level = lev;
            imgLev = pyramid.GetLevelImage(lev)

            if (lev == top):
                (C, sigma) = self.__boot.Fit(imgLev, np.zeros((3,3)), float('inf'))
                print("\a"); cv.waitKey(0)
                if np.array_equal(C, LevelEstimator.EmptyConicMatrix()):
                    sigma = 0
                    return LevelEstimator.GetGeometry(C)

            if np.array_equal(C, LevelEstimator.EmptyConicMatrix())==False:
                (C, sigma) = self.__optim.Fit(imgLev, C, sigma)
                print("\a"); cv.waitKey(0)

            self.__edge.CannyThreshold *= 1.1;
        
        Cden = self.__optim.DenormalizeConic(C)
        return LevelEstimator.GetGeometry(Cden/Cden[2,2])

    @property
    def EdgePoints(self):
        return self.__optim.RawPoints

    @property
    def Weights(self):
        return self.__optim._weights
