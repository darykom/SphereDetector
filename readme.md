# Sphere Detector

**Author:** Dario Comanducci  


## Description

This Python project implements a system for the accurate detection and estimation of a sphere in digital images (a first step for a photometric-stereo application, where the sphere acts as a calibration device).  
It uses a multi-resolution approach (Gaussian pyramid) and several robust optimization techniques to refine the sphere localization and parameters, starting from an initialization based on the Hough transform for circles.
Mathematical explanation is provided in ./docs/SphereDetector.pdf, while UML class diagram is available in ClassDiagram.pdf.

## Repository Structure
SphereDetector/  
│  
├── src/  
│   ├── __init__.py  
│   ├── EdgeExtractor.py  
│   ├── EstimationMode.py  
│   ├── Kickstart.py  
│   ├── LevelEstimator.py  
│   ├── LevelEstimatorCreator.py  
│   ├── LevelOptimizer.py  
│   ├── MAIN_SphereDetector.py  
│   ├── Observer.py  
│   ├── PyramidOfGaussians.py  
│   ├── Settings.py  
│   ├── SphereDetector.py  
│   ├── Subject.py  
│   └── Viewer.py  
│  
├── data/  
│   ├── DSC_3112.JPG  
│   └── DSC_5165.JPG  
│  
├── docs/  
│   ├── SphereDetector.pdf  
│   └── ClassDiagram.pdf  
│  
├── outputs/  some images saved manually by OpenCV window  
│  
├── requirements.txt  
└── README.md  

### Class diagram  

```mermaid
classDiagram
    class EstimationMode {
        <<enum>>
        HOUGH
        LS
        HUBER
        L1L2
        GERMANMCCLURE
        TUKEY
    }

    class EdgeExtractor {
        +SetRoi(C: np.array)
        +Extract(gray: np.array) list((float, float))
        +CannyThreshold: float
        -int __canny1
        -int __canny2
        -tuple __roi
    }

    class PyramidOfGaussians {
        +PyramidOfGaussians(image, levels)
        +Levels: int
        +GetLevelImage(level)
        -int __levels
        -__pyramid: list
        -__T: list
        -__BuildPyramid(image)
    }

    class Subject {
        +Attach(observer: Observer)
        +Detach(observer: Observer)
        +Notify()
    }

    class Observer {
        <<abstract>>
        +Update(changedSubject: Subject)
    }

    class SFpaletteOCV {
        +SFpaletteOCV()
        +CreatePalette() list
        +FewLightBrown: tuple
        +FewMediumBrown: tuple
        +FewDarkBrown: tuple
        +FewLightGreen: tuple
        +FewMediumGreen: tuple
        +FewDarkGreen: tuple
        +FewLightOrange: tuple
        +FewMediumOrange: tuple
        +FewDarkOrange: tuple
        +FewLightBlue: tuple
        +FewMediumBlue: tuple
        +FewDarkBlue: tuple
        +FewLightPurple: tuple
        +FewMediumPurple: tuple
        +FewDarkPurple: tuple
        +FewLightYellow: tuple
        +FewMediumYellow: tuple
        +FewDarkYellow: tuple
    }

    class Viewer {
        +Viewer()
        +Update(changedSubject: Subject)
        +Display(title: str, image: np.array, C: np.array, points: list, res: list, weights: list)
        -Palette: list
    }

    class LevelEstimator {
        +TotalIterations: int
        +DeltaC: float
        +Method: str
        +Level: int
        +Image: np.array
        +C: np.array
        +Sigma: float
        +Distances: list
        +Fit(image: np.array, C: np.array, sigma: float) (np.array, float)
        +SetWeights(sigma)
        +NormalizingTransformation(image: np.array) np.array
        +ComputeWeight(res: (float, float), sigma: float, thr)
        +NormalizePoints(q:list((float, float)), T:np.array) list((float, float))
        +EvaluateScale(residuals:list) float
        +RawPoints: list
        +DenormalizeConic(C: np.array) np.array
        +DenormalizePoint(q:(float, float), Tnormalizing:np.array) (float, float)
        +DenormalizePoints(q:list((float, float)), Tnormalizing:np.array) list((float, float))
        +GetRoi(C: np.array) ((int, int), (int, int))
        +EmptyConicMatrix()
        +BuildConicMatrix(p)
        +GetGeometry(C: np.array) (float, float, float, float, float)
        +CheckConvergence(C1: np.array, C2: np.array) (float, bool)
        #EdgeExtractor _edges
        #_Points: list
        #_distances: list
        #_weights: list
        #_T: np.array
        #_convergenceThreshold: float
        #_maxIterations: int
        #_image: np.array
        #_C: np.array
        #_sigma: float
        #_ComputeGrossOutlierThreshold(C: np.array, sigma: float) float
        #_ComputeGeometricThreshold(C: np.array) float
        #_FitCore(image: np.array, C:np.array, sigma: float) np.array
        #_ComputeDistances(C:np.array)
        #_SetOutlierThreshold(sigma: float) float
        -EstimationMode __estimationMode
        -int __level
        -__RawPoints: list
        -__ExtractPoints(image: np.array, C: np.array)
        -__SetNormalizingTransformation(image: np.array)
        -__NormalizePoints()
        -__EvaluateResidualScale(C:np.array, thr:float)
    }

    class Kickstart {
        +X: float
        +Y: float
        +R: float
        +Kickstart(estimMode: EstimationMode, edgeExtractor: EdgeExtractor)
        +_FitCore(image: np.array, C:np.array, sigma: float) np.array
        +_ComputeGeometricThreshold(C:np.array) float
        +SetWeights(sigma)
    }

    class LevelOptimizer {
        +LevelOptimizer(estimMode: EstimationMode, edgeExtractor: EdgeExtractor)
        +ComputeWeight(res, sigma, thr)
        +_FitCore(image: np.array, C: np.array, sigma: float) np.array
        +__BuildConicMatrix(p:(float, float, float, float, float, float)) np.array
        +__SolveSystem(M:np.array, S:np.array) (float, float, float, float, float, float)
        +_ComputeGeometricThreshold(C: np.array) float
        -np.array __M
        -__weightFunction
        -__LS(r: float) float
        -__Huber(r: float) float
        -__L1L2(r: float) float
        -__GermanMcClure(r: float) float
        -__Tukey(r:float) float
        -__ComposeSMatrix() np.array
    }

    class LevelEstimatorCreator {
        +Factory(estimMode: EstimationMode, edgeExtractor: EdgeExtractor) LevelEstimator
    }

    class SphereDetector {
        +SphereDetector(estimMode: EstimationMode)
        +Viewer: Viewer
        +Localize(image: np.array, levels: int) (float, float, float, float, float)
        +EdgePoints: list
        +Weights: list
        -EdgeExtractor __edge
        -LevelEstimator __boot
        -LevelEstimator __optim
        -Viewer __viewer
    }

    Subject <|-- LevelEstimator
    Observer <|-- Viewer
    Viewer o-- SFpaletteOCV
    LevelEstimator <|-- Kickstart
    LevelEstimator <|-- LevelOptimizer
    LevelEstimatorCreator ..> LevelEstimator : creates
    LevelEstimatorCreator ..> Kickstart : creates
    LevelEstimatorCreator ..> LevelOptimizer : creates
    SphereDetector o-- PyramidOfGaussians
    SphereDetector o-- EdgeExtractor
    SphereDetector o-- LevelEstimatorCreator
    SphereDetector o-- LevelEstimator
    SphereDetector o-- Viewer
    LevelEstimator o-- EdgeExtractor
    LevelOptimizer o-- EstimationMode
    Kickstart o-- EstimationMode
```
