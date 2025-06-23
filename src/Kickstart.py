from LevelEstimator import LevelEstimator
from EstimationMode import EstimationMode
from EdgeExtractor import EdgeExtractor

import cv2 as cv
import numpy as np
#from enum import Enum

class Kickstart(LevelEstimator): #Kickstart Ignition
    """Initialization by Hough Transform"""
    X: float
    Y: float
    R: float

    def __init__(self, estimMode: EstimationMode, edgeExtractor: EdgeExtractor):
        LevelEstimator.__init__(self, estimMode, edgeExtractor)
        self._maxIterations = 1

    def _FitCore(self, image: np.array, C:np.array, sigma: float) -> np.array:
        #img = np.zeros(image.shape)
        #cv.erode(image, cv.getStructuringElement(cv.MORPH_ERODE, 3), dst=img)
        # Non opera sui punti (x,y) ma sull'immagine, ha parametri specifici di funzionamento...
        rows, cols = map(int, image.shape)
        minDist = min(rows, cols)-1
        resolution = 1
        canny = self._edges.CannyThreshold
        circleList = cv.HoughCircles(image=image, method=cv.HOUGH_GRADIENT, dp=resolution, minDist=minDist, param1=canny, param2=16, minRadius=8, maxRadius=0)

        if len(circleList)==0:
            Cnew = LevelEstimator.EmptyConicMatrix()
        else:
            circle = circleList[0][0]
            # parametri del cerchio in pixel
            xi = circle[0]
            yi = circle[1]
            ri = circle[2]

            

            # I valori in uscita dalla funzione cv.HoughCircles sono espressi in coordinate pixel, vanno normalizzate
            # Per evitare di invertire T nella mappatura della conica, trasformo direttamente le coordinate del centro
            # c' = T*ci, mentre per il raggio sfrutto r' = T*[ri; 0; 1]-T*[0; 0; 1] = T*[r; 0; 0] = T[0,0]*r
            c = self._T @ np.array([xi, yi, 1]); xc = c[0]; yc=c[1]
            r = self._T[0,0]*ri
            f = xc*xc + yc*yc - r*r
            Cnew = np.array([[  1,   0, -xc],
                             [  0,   1, -yc], 
                             [-xc, -yc,   f]])

            Kickstart.X = xc; Kickstart.Y=yc; Kickstart.R=r
            self._ComputeDistances(Cnew)

        return Cnew


    def _ComputeGeometricThreshold(self, C:np.array) -> float:
        geom = self.GetGeometry(C)
        return max(geom[2], geom[3])/3

    def SetWeights(self, sigma):
        thr = self._ComputeGrossOutlierThreshold(self.C, sigma)
        self._weights = []
        for n in range(len(self._distances)):
            weight = self.ComputeWeight(self._distances[n], sigma, thr)
            self._weights.append(weight)

        
