from enum import Enum, auto

class EstimationMode(Enum):
    HOUGH  = auto() # Hough transform with circles
    LS  = auto() # Straight least square method
    HUBER = auto() # Weighted least square by Huber function
    L1L2 = auto() # Weighted least square by L1-L2 function
    GERMANMCCLURE = auto() # Weighted least square by GermanMcClure function
    TUKEY = auto() # Weighted least square by Beaton - Tukey function


