from EstimationMode import EstimationMode
from LevelEstimator import LevelEstimator
from Kickstart import Kickstart
from LevelOptimizer import LevelOptimizer
from EdgeExtractor import EdgeExtractor



class LevelEstimatorCreator(object):
    """description of class"""

    @staticmethod
    def Factory(estimMode: EstimationMode, edgeExtractor: EdgeExtractor) ->LevelEstimator:
        if estimMode==EstimationMode.HOUGH: return Kickstart(EstimationMode.HOUGH, edgeExtractor)
        return LevelOptimizer(estimMode, edgeExtractor)


