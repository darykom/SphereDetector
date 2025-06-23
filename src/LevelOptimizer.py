from EstimationMode import EstimationMode
from LevelEstimator import LevelEstimator
from EdgeExtractor import EdgeExtractor

import math
import cv2 as cv
import numpy as np
import scipy.linalg as scLA

class LevelOptimizer(LevelEstimator):
    """description of class"""

    def __LS(self, r: float) -> float:
        if math.fabs(r)<3: return 1
        return 0

    def __Huber(self, r: float) -> float:
        k = 1.345; absr = math.fabs(r)
        if absr<=k: return 1
        return k/absr

    def __L1L2(self, r: float) -> float:
        return 1/math.sqrt(1+(r*r)/2)

    def __GermanMcClure(self, r: float) -> float:
        v = 1 + r*r
        return 1/(v*v);

    def __Tukey(self, r:float) -> float:
        k = 4.685; absr = math.fabs(r)
        if absr>k: return 0
        r_k = r/k; d = 1 - r_k*r_k;
        return d*d


    def __init__(self, estimMode: EstimationMode, edgeExtractor: EdgeExtractor):

        LevelEstimator.__init__(self, estimMode, edgeExtractor)

        self.__M = np.zeros((6,6))
        self.__M[0,2] = self.__M[2,0]  = 2
        self.__M[1,1] = -1

        if (estimMode == EstimationMode.LS): self.__weightFunction = self.__LS
        if (estimMode == EstimationMode.HUBER): self.__weightFunction = self.__Huber
        if (estimMode == EstimationMode.TUKEY): self.__weightFunction = self.__Tukey
        if (estimMode == EstimationMode.L1L2): self.__weightFunction = self.__L1L2
        if (estimMode == EstimationMode.GERMANMCCLURE): self.__weightFunction = self.__GermanMcClure
        if (estimMode == EstimationMode.HOUGH): raise ValueError("Hough transform not allowed for refinement")

    def ComputeWeight(self, res, sigma, thr):
        #if thr == 0:
        #    thr = LevelEstimator._SetOutlierThreshold(sigma)
        if math.fabs(res) > thr:
            return -1
        dnorm = res/sigma
        weight = self.__weightFunction(dnorm)
        return weight

    def _FitCore(self, image: np.array, C: np.array, sigma: float) -> np.array: 

        self._ComputeDistances(C)
        self.SetWeights(sigma)
        S = self.__ComposeSMatrix()

        p = LevelOptimizer.__SolveSystem(self.__M, S)
        Cnew = LevelEstimator.BuildConicMatrix(p)

        return Cnew


    @staticmethod
    def __BuildConicMatrix(p:(float, float, float, float, float, float)) -> np.array:
        """
        DESCRIPTION: It constructs the matrix representation of the Conic from the parameter vector p
        INPUT: 
        - p: array with parameters for the conic equation p(0) x^2 + p(1) xy + p(2) y^2 + p(3) x + p(4) y + p(5)=0
        OUTPUT:
        - Cn: homgeneous matrix representation of the conic
        """
        Cn = np.array([[p[0],   p[1]/2, p[3]/2], 
                       [p[1]/2, p[2],   p[4]/2], 
                       [p[3]/2, p[4]/2, p[5]]])
        return Cn


    def __ComposeSMatrix(self) -> np.array:
        """
        DESCRIPTION: It constructs the matrices for the eigenvalue problem Sp = λ Mp
        INPUT: 
        - x, y: arrays of points coordinates
        OUTPUT:
        - s: matriX for the eigenvalue problem Sp = λ Mp
        """
        x, y = zip(*(self._Points))
        D = np.empty(shape=[0, 6])
        w = []
        for n in range(len(self._weights)):
            weight = self._weights[n]
            if (weight>0):
                w.append(weight)
                D = np.vstack((D, [x[n]*x[n], x[n]*y[n], y[n]*y[n], x[n], y[n], 1 ]))
        W = np.diag(w)
        S = D.T @ W @ D
        return S


    @staticmethod
    def __SolveSystem(M:np.array, S:np.array) -> (float, float, float, float, float, float):
        """
        DESCRIPTION: It solves the "inverse" eigenvalue problem Mp = 1/λ S p
        INPUT: 
        - M, S: matrices for the eigenvalue problem Mp = 1/λ S p
        OUTPUT:
        - p: array solution of Mp = 1/λ S p (parameters for the conic equation p(0) x^2 + p(1) xy + p(2) y^2 + p(3) x + p(4) y + p(5)=0 constrained to p(1)^2-4p(0)p(2)<0)
        """

        eigvals, eigvecs = scLA.eigh(M, S) #ok
        #eigvals, eigvecs = scLA.eigh(S, M) ######## ERRORE perché M non è def. pos.

        #E = scLA.solve(S,M, assume_a='sym') # Oss: E = S^{-1} M NON è matrice simmetrica, anche se S^{-1} e M sono simmetriche
        #eigvals, eigvecs = np.linalg.eig(E) #ok
        #eigvals, eigvecs = scLA.eig(E) #ok
        #eigvals, eigvecs = scLA.eigh(E) ######## NON FUNZIONA perché E non è simmetrica
        #eigvals, eigvecs = np.linalg.eigh(E) ######## NON FUNZIONA perché E non è simmetrica
        
        # altro metodo con Fattorizzazione di Cholesky (cfr. Numerical Recipies 3rd ed. pp 568-569)
        #L = scLA.cholesky(S).T
        #Y = scLA.solve(L,M.T)
        #X = scLA.solve(L,Y.T) # X = L^{-1}ML{-T} è matrice simmetrica: X^T = X
        #eigvals, eigvecs = scLA.eigh(X) #ok perché X simmetrica
        #eigvals, eigvecs = scLA.eig(X) #ok
        #eigvals, eigvecs = np.linalg.eig(X) #ok
        #eigvals, eigvecs = np.linalg.eigh(X) #ok perché X simmetrica
        
        miId = np.argmax(eigvals)
        mi = eigvals[miId]
        p = eigvecs[:,miId] # se uso scLA.eigh(M, S) oppure np.linalg.eig(E) o scLA.eig(E)
        #p = scLA.solve(L.T, eigvecs[:,miId]) #se uso scLA.eigh(X)
        
        # di solito ridondante, serve solo a soddisfare il vincolo 4ac-b^2=1 (ma per un ellisse basta b^2-4ac<0)
        # tuttavia qui necessario per stabilire la convergenza nell'approccio iterativo, fissando il fattore di scala
        scal = 1/np.sqrt(p.T@(M@p))
        p = scal*p

        # output sulla shell
        #print("mi: ", eigvals)
        #print("miId: ", miId)        
        #print("lambda: ", 1/mi)
        #print("Sp - lambda Mp: ", S @ p - 1/mi * M@p)
        #print("p.T M p: ", p.T@(M@p))
        #print("p: ",  p)
        #print("norm(p): ",  math.sqrt(p.T@p))

        return tuple(p.tolist())

    def _ComputeGeometricThreshold(self, C: np.array) -> float:
        if np.array_equal(C, LevelEstimator.EmptyConicMatrix()):
            return float('inf')
        geom = self.GetGeometry(C)
        return max(geom[2], geom[3])/16



