import cv2 as cv

from SphereDetector import SphereDetector
from SphereDetector import EstimationMode

from Settings import PYRAMID_LEVELS


if __name__ == "__main__":
    img = cv.imread('../data//DSC_3112.JPG')
    # img = cv.imread('../data//DSC_5165.JPG')
    
    
    # sphere = SphereDetector(EstimationMode.LS)
    sphere = SphereDetector(EstimationMode.HUBER)
    # sphere = SphereDetector(EstimationMode.L1L2)
    # sphere = SphereDetector(EstimationMode.TUKEY)
    # sphere = SphereDetector(EstimationMode.GERMANMCCLURE)
    ellipse = sphere.Localize(img, PYRAMID_LEVELS)
    
    cv.destroyAllWindows()



