import math
import cv2 as cv
import numpy as np

from abc import abstractmethod
from EstimationMode import EstimationMode
from EdgeExtractor import EdgeExtractor
from Subject import Subject #, abstractmethod

from Settings import CONVERGENCE_THRESHOLD, MAX_ITERATIONS

class LevelEstimator(Subject):
    """description of class"""

    TotalIterations = 0
    DeltaC = float("inf")

    def __init__(self, estimMode: EstimationMode, edgeExtractor: EdgeExtractor):
        
        Subject.__init__(self)
        self.__estimationMode = estimMode
        self.__level = None
        self._edges = edgeExtractor
        self.__RawPoints = []
        self._Points = []
        self._distances = []
        self._weights = []
        self._T = np.eye(3)
        self._convergenceThreshold = CONVERGENCE_THRESHOLD
        self._maxIterations = MAX_ITERATIONS
        self._image = np.array([]);
        self._C = LevelEstimator.EmptyConicMatrix()
        self._sigma = float('inf')

    @property
    def Method(self):
        return self.__estimationMode.name;

    @property
    def Level(self):
        return self.__level;
    @Level.setter
    def Level(self, level: int):
        self.__level = level

    @property
    def Image(self):
        return self._image
    @Image.setter
    def Image(self, image: np.array):
        self._image = image

    @property
    def C(self):
        return self._C;

    @property
    def Sigma(self):
        return self._sigma;

    @property
    def Distances(self):
        return self._distances;


    def Fit(self, image: np.array, C: np.array, sigma:float) -> (np.array, float):
        #from SphereDetector import SphereDetector
        self.Image = image;
        self.__SetNormalizingTransformation(image)

        for it in range(self._maxIterations):
            LevelEstimator.TotalIterations += 1;

            Cden = self.DenormalizeConic(C)
            self.__ExtractPoints(image, Cden)
            self.__NormalizePoints()

            # sono costretto a passare in FitCore l'immagine e non i punti perché Kickstart.FitCore impiega cv.HoughCircle(image,...)
            # le altre classi derivate invece lavorano su self._points e self._distances confrontati con sigma
            CNew = self._FitCore(image, C, sigma) 

            thr = self._ComputeGrossOutlierThreshold(CNew, sigma)
            sigmaNew = self.__EvaluateResidualScale(CNew, thr)

            LevelEstimator.DeltaC, conv = self.CheckConvergence(C, CNew)

            self._C = CNew
            self._sigma = sigmaNew
            
            self.Notify()
            #print('{0}\t{1}\t{2}\t| {3} {4}'.format(self.__level, it+1, LevelEstimator.TotalIterations, LevelEstimator.GetGeometry(CNew/CNew[0,0]), DeltaC))

            if conv:
                return (CNew, sigmaNew)

            C = CNew
            sigma = sigmaNew

        return (CNew, sigmaNew)


    def _ComputeGrossOutlierThreshold(self, C: np.array, sigma:float) -> float:
            thrRad = self._ComputeGeometricThreshold(C)
            thr = min(thrRad, LevelEstimator._SetOutlierThreshold(sigma))
            return thr

    @abstractmethod
    def _ComputeGeometricThreshold(C: np.array) -> float:
        pass

    def SetWeights(self, sigma):
        thr = self._ComputeGrossOutlierThreshold(self.C, sigma) #LevelEstimator._SetOutlierThreshold(sigma)
        self._weights = []
        for n in range(len(self._distances)):
            weight = self.ComputeWeight(self._distances[n], sigma, thr)
            self._weights.append(weight)


    def __ExtractPoints(self, image: np.array, C: np.array):
        self._edges.SetRoi(C)
        self.__RawPoints = self._edges.Extract(image)

    def __SetNormalizingTransformation(self, image: np.array) -> None:
        self._T = LevelEstimator.NormalizingTransformation(image)

    @staticmethod
    def NormalizingTransformation(image: np.array) -> np.array:
        ymax, xmax = image.shape;
        ymax -= 1; xmax -=1;
        s = max(ymax, xmax)
        T = np.array([[2/s,   0, -xmax/s], 
                      [  0, 2/s, -ymax/s], 
                      [  0,   0,       1]])
        return T

    def __NormalizePoints(self):
        self._Points = LevelEstimator.NormalizePoints(self.__RawPoints, self._T)

    @abstractmethod
    def _FitCore(self, image: np.array, C:np.array, sigma: float) -> np.array: 
        pass

    def ComputeWeight(self, res:(float, float), sigma:float, thr):
        if math.fabs(res)<thr:
            return 1
        return -1


    @staticmethod
    def NormalizePoints(q:list((float, float)), T:np.array) -> list((float, float)):
        x, y = zip(*q)
        barq =  T @ np.array([x, y, [1] * len(x)])
        return list(zip(barq[0,:].tolist(), barq[1,:].tolist()))

    

    def _ComputeDistances(self, C:np.array):
        self._distances = LevelEstimator.SampsonDistance(C, self._Points)
        
    

    def __EvaluateResidualScale(self, C:np.array, thr:float):
        distances = []
        for n in range(len(self._distances)):
            if math.fabs(self._distances[n]) <= thr:
                newdist = LevelEstimator.SampsonDistance(C, [self._Points[n]])
                distances.append(newdist[0])
        sigma = LevelEstimator.EvaluateScale(distances)
        return sigma

    @staticmethod
    def _SetOutlierThreshold(sigma: float) -> float:
        return 6*sigma

    @staticmethod
    def SampsonDistance(C:np.array, q:list((float, float))) -> list:
        x, y = list(zip(*q))
        sampson = []
        N = len(x)
        for n in range(0,N):
            q = np.array( [x[n], y[n], 1])
            v = C @ q
            num = q.T @ v
            xv = v[0]; yv = v[1]
            den = 2 * math.sqrt(xv*xv + yv*yv)
            dst = num/den
            sampson.append(dst)
        return sampson


    @staticmethod
    def EvaluateScale(residuals:list) -> float:
        N = len(residuals); p = 5; 
        if p>=N: p = 0;

        res2 = []
        for n in range(0,N):
            res2.append(residuals[n]*residuals[n])
        med = np.sqrt(np.median(res2))
        s1 = 1.4826 * (1+5/(N-p))*med
        thr = s1*2.5
        thr2 = thr*thr

        num = 0
        den = -p
        for r2 in res2:
            if (r2<thr2):
                num += r2
                den += 1

        return math.sqrt(num/den)

    @property
    def RawPoints(self):
        return self.__RawPoints

    def DenormalizeConic(self, C: np.array) -> np.array:
        return self._T.T @ C @ self._T

    # def DenormalizePoint(self, q:(float, float)) -> (float, float):
    #     return LevelEstimator.DenormalizePoint(q, self._T)


    @staticmethod
    def DenormalizePoint(q:(float, float), Tnormalizing:np.array) -> (float, float):
        x, y = q
        p = np.array([x, y, 1])
        q = np.linalg.solve(Tnormalizing,p)
        return (q[0]/q[2], q[1]/q[2])

    @staticmethod
    def DenormalizePoints(q:list((float, float)), Tnormalizing:np.array) -> list((float, float)):
        pts = []
        N = len(q)
        for n in range(N):
            p = LevelEstimator.DenormalizePoint(q[n],Tnormalizing)
            pts.append(p)
        return pts


    @staticmethod
    def GetRoi(C: np.array) -> ((int, int), (int, int)):
        uCentre, vCentre, Ru, Rv, thetarad = LevelEstimator.GetGeometry(C)
        rRoi = np.round(1.2*max(Ru, Rv));

        topLeftRoi = (max(0,uCentre-rRoi), max(0,vCentre-rRoi))
        bottomRightRoi = (uCentre+rRoi, vCentre+rRoi)
        return (topLeftRoi, bottomRightRoi)

    @staticmethod
    def EmptyConicMatrix():
        return np.zeros((3,3))


    @staticmethod
    def BuildConicMatrix(p):
        """
        DESCRIPTION: It constructs the matrix representation of the Conic from the parameter vector p
        INPUT: 
        - p: array with parameters for the conic equation p(0) x^2 + p(1) xy + p(2) y^2 + p(3) x + p(4) y + p(5)=0
        OUTPUT:
        - Cn: homgeneous matrix representation of the conic
        """
        C = np.array([[p[0],   p[1]/2, p[3]/2], 
                      [p[1]/2, p[2],   p[4]/2], 
                      [p[3]/2, p[4]/2, p[5]]])
        return C


    @staticmethod
    def GetGeometry(C: np.array) -> (float, float, float, float, float):

        if np.array_equal(C, LevelEstimator.EmptyConicMatrix()):
            return (0, 0, 0, 0, 0)

        #Matlab code
        #thetarad = 0.5*atan2(par(2),par(1) - par(3));
        #cost = cos(thetarad);
        #sint = sin(thetarad);
        #sin_squared = sint.*sint;
        #cos_squared = cost.*cost;
        #cos_sin = sint .* cost;

        par = (C[0,0], C[0,1]*2, C[1,1], C[0,2]*2, C[1,2]*2, C[2,2])

        thetarad = 0.5* math.atan2(par[1],par[0] - par[2])
        cost = math.cos(thetarad)
        sint = math.sin(thetarad)
        sin_squared = sint*sint
        cos_squared = cost*cost
        cos_sin = sint * cost

        #Matlab code
        #Ao = par(6);
        #Au =   par(4) .* cost + par(5) .* sint;
        #Av = - par(4) .* sint + par(5) .* cost;
        #Auu = par(1) .* cos_squared + par(3) .* sin_squared + par(2) .* cos_sin;
        #Avv = par(1) .* sin_squared + par(3) .* cos_squared - par(2) .* cos_sin;
        Ao = par[5]
        Au =   par[3] * cost + par[4] * sint
        Av = - par[3] * sint + par[4] * cost
        Auu = par[0] * cos_squared + par[2] * sin_squared + par[1] * cos_sin
        Avv = par[0] * sin_squared + par[2] * cos_squared - par[1] * cos_sin

        #Matlab code
        #% ROTATED = [Ao Au Av Auu Avv]
        #tuCentre = - Au./(2.*Auu);
        #tvCentre = - Av./(2.*Avv);
        #wCentre = Ao - Auu.*tuCentre.*tuCentre - Avv.*tvCentre.*tvCentre;
        #uCentre = tuCentre .* cost - tvCentre .* sint;
        #vCentre = tuCentre .* sint + tvCentre .* cost;
        tuCentre = - Au/(2*Auu)
        tvCentre = - Av/(2*Avv)
        wCentre = Ao - Auu*tuCentre*tuCentre - Avv*tvCentre*tvCentre
        uCentre = tuCentre * cost - tvCentre * sint
        vCentre = tuCentre * sint + tvCentre * cost

        #Matlab code
        #Ru = -wCentre./Auu;
        #Rv = -wCentre./Avv;
        #Ru = sqrt(abs(Ru)).*sign(Ru);
        #Rv = sqrt(abs(Rv)).*sign(Rv);
        Ru = -wCentre/Auu
        Rv = -wCentre/Avv
        Ru = math.sqrt(abs(Ru))*np.sign(Ru)
        Rv = math.sqrt(abs(Rv))*np.sign(Rv)

        #a = [uCentre, vCentre, Ru, Rv, thetarad];
        #if (thetarad<0):
        #    thetarad = np.pi/2-thetarad
        #    Ru, Rv = Rv, Ru
        return (uCentre, vCentre, Ru, Rv, thetarad)

    def CheckConvergence(self, C1: np.array, C2: np.array) -> (float, bool):
        for r in range(3):
            for c in range(3):
                if C1[r,c]*C2[r,c]<0:
                    C1 = -C1
                    break
        
        dist = np.linalg.norm(C1-C2)
        return (dist, dist<self._convergenceThreshold)

        g1 = LevelEstimator.GetGeometry(C1);
        xc1 = g1[0]
        yc1 = g1[1]
        ra1 = g1[2]
        rb1 = g1[3]
        thetaRad1 = g1[4]
        R1 = np.array([[math.cos(thetaRad1), -math.sin(thetaRad1), 0], 
                       [math.sin(thetaRad1),  math.cos(thetaRad1), 0],
                       [                  0,                    0, 1]])
        a1 = R1 @ np.array([ra1,   0, 1]); a1 = a1/a1[2];
        b1 = R1 @ np.array([  0, rb1, 1]); b1 = b1/b1[2];

        g2 = LevelEstimator.GetGeometry(C2);
        xc2 = g2[0]
        yc2 = g2[1]
        ra2 = g2[2]
        rb2 = g2[3]
        thetaRad2 = g2[4]
        #if (thetaRad1*thetaRad2<0):
        #    ra2, rb2 = rb2, ra2
        R2 = np.array([[math.cos(thetaRad1), -math.sin(thetaRad1), 0], 
                       [math.sin(thetaRad1),  math.cos(thetaRad1), 0],
                       [                  0,                    0, 1]])
        a2 = R2 @ np.array([ra2,   0, 1]); a2 = a2/a2[2];
        b2 = R2 @ np.array([  0, rb2, 1]); b2 = b2/b2[2];
        deltaA = a1-a2
        deltaB = b1-b2
        deltaC = np.array([xc1-xc2, yc1-yc2, 0])
        delta = np.linalg.norm(deltaA) + np.linalg.norm(deltaB) + np.linalg.norm(deltaC)
        return (delta, delta < self._convergenceThreshold)

