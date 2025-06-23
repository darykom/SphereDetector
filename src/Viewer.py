import math
import numpy as np
import cv2 as cv

from Observer import Observer 
#from Subject import Subject
from LevelEstimator import LevelEstimator
from Kickstart import Kickstart

from Settings import VIEWER_WIDTH


# FewLightBrown = (84, 165, 191)
# FewMediumBrown = (47, 145, 178)
# FewDarkBrown = (42, 114, 157)

# FewLightGreen = (151, 205, 144)
# FewMediumGreen = (104, 189, 96)
# FewDarkGreen = (72, 151, 5)

# FewLightOrange = (88, 178, 251)
# FewMediumOrange = (58, 164, 250)
# FewDarkOrange = (36, 92, 223)

# FewLightBlue = (230, 189, 136)
# FewMediumBlue = (218, 165, 93)
# FewDarkBlue = (171, 93, 38)

# FewLightPurple = (199, 153, 188)
# FewMediumPurple = (178, 118, 178)
# FewDarkPurple = (150, 58, 123)

# FewLightYellow = (70, 221, 237)
# FewMediumYellow = (63, 207, 222)
# FewDarkYellow = (46, 180, 199)

# FewLightRed = (110, 126, 240)
# FewMediumRed = (84, 88, 241)
# FewDarkRed = (39, 32, 203)

class SFpaletteOCV:
	LightGray   = (140, 140, 140)
	LightBlue   = (230, 189, 136)
	LightOrange = ( 88, 178, 251)
	LightGreen  = (151, 205, 144)
	LightPink   = (201, 170, 246)
	LightBrown  = ( 84, 165, 191)
	LightPurple = (199, 153, 188)
	LightYellow = ( 70, 221, 237)
	LightRed    = (110, 126, 240)
	Gray        = ( 77,  77,  77)
	Blue        = (218, 165,  93)
	Orange      = ( 58, 164, 250)
	Green       = (104, 189,  96)
	Pink        = (176, 124, 241)
	Brown       = ( 47, 145, 178)
	Purple      = (178, 118, 178)
	Yellow      = ( 63, 207, 222)
	Red         = ( 84,  88, 241)
	DarkGray    = ( 38,  38,  38)
	DarkBlue    = (171,  93,  38)
	DarkOrange  = ( 36,  92, 223)
	DarkGreen   = ( 72, 151,   5)
	DarkPink    = (111,  18, 229)
	DarkBrown   = ( 42, 114, 157)
	DarkPurple  = (150,  58, 123)
	DarkYellow  = ( 46, 180, 199)
	DarkRed     = ( 39,  32, 203)


class Viewer(Observer):
    """description of class"""

    #def __init__(self, levelEstimator: LevelEstimator):
        #self.__subject = levelEstimator
    def __init__(self):
        self.__iterations = []
        self.__deltaC = []
        self.__sigma = []
        self.__level = []
    
    @property
    def HistoryIterations(self):
        return self.__iterations

    @property
    def HistoryDeltaC(self):
        return self.__deltaC

    @property
    def HistorySigma(self):
        return self.__sigma

    @property
    def HistoryLevel(self):
        return self.__level


    def Update(self, changedSubject: LevelEstimator): 

        if changedSubject.Method == "HOUGH":
            changedSubject.SetWeights(changedSubject.Sigma)

        self.__iterations.append(LevelEstimator.TotalIterations)
        self.__deltaC.append(LevelEstimator.DeltaC)
        self.__sigma.append(changedSubject.Sigma)
        self.__level.append(changedSubject.Level)

        img = changedSubject.Image
        dimy = VIEWER_WIDTH; dimx = math.floor(dimy*img.shape[1]/img.shape[0]);
        imgResized = cv.resize(img, (dimx, dimy), interpolation = cv.INTER_NEAREST)
        #imgResized = cv.resize(img, (dimx, dimy), interpolation = cv.INTER_CUBIC)
        Tres = LevelEstimator.NormalizingTransformation(imgResized)
        Cresized = Tres.T @ changedSubject.C @ Tres
        ptsResized = LevelEstimator.DenormalizePoints(changedSubject._Points, Tres)

        xr, yr = LevelEstimator.DenormalizePoint((changedSubject.Sigma, 0), Tres)
        xo, yo = LevelEstimator.DenormalizePoint((0, 0), Tres)
        dx = xr-xo; dy = yr-yo
        sigmaResized = math.sqrt(dx*dx + dy*dy)
        
        bootCenter = LevelEstimator.DenormalizePoint((Kickstart.X, Kickstart.Y), Tres)
        xr, yr = LevelEstimator.DenormalizePoint((Kickstart.R*1.5, 0), Tres)
        dx = xr-xo; dy = yr-yo
        r = math.ceil(math.sqrt(dx*dx + dy*dy)/2)*2
        textCorner = (math.ceil(bootCenter[0]+1.5*r), math.ceil(bootCenter[1]-r/2))
        barCorner = (math.ceil(bootCenter[0]+1.5*r), math.ceil(bootCenter[1]-r/2-100))

        
        #text = []
        #text.append("it.={}".format(LevelEstimator.TotalIterations))
        #text.append("it.={}".format(LevelEstimator.TotalIterations))

        #samps = changedSubject.Distances
        #f = open('SampsonInitLev4.dat', 'w')
        #for n in range(0, len(samps)):
        #    line = '{0} {1} {2}\n'.format(xRaw[n], yRaw[n], samps[n])
        #    f.write(line)
        #f.close()
        #cv.imwrite('imgLev4.png', img)


        title = 'Sphere Detector'
        Viewer.Draw(title, changedSubject.Method, changedSubject.Level, imgResized, Cresized, ptsResized, changedSubject._weights, sigmaResized, LevelEstimator.DeltaC, textCorner, barCorner, img.shape)


    @staticmethod
    def Draw(title, method, level, img, C, points, weights, scale, deltaC, textCorner, barCorner, imgSize):
        imgDemo = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        geom = LevelEstimator.GetGeometry(C/C[2,2])
        center = (geom[0], geom[1])
        box = (geom[2]*2, geom[3]*2)
        angle = geom[4]*180/np.pi
        iter = LevelEstimator.TotalIterations

        #xtext, ytext = center-round(max(box)/2)
        #xtext = int(np.round(center[0]-max(box)/1.1))
        #ytext = int(np.round(center[1]-4*min(box)/2))
        xtext = textCorner[0]
        ytext = textCorner[1]
        htext = 20

        cv.putText(imgDemo,("{}").format(method),         (xtext,ytext),         cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("lev.= {} ({}x{})").format(level, imgSize[0], imgSize[1]),    (xtext,ytext+  htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("it. = {}").format(iter),     (xtext,ytext+2*htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("x  = {}").format(center[0]), (xtext,ytext+3*htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("y  = {}").format(center[1]), (xtext,ytext+4*htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("a  = {}").format(max(box)),  (xtext,ytext+5*htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("b  = {}").format(min(box)),  (xtext,ytext+6*htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("<) = {}").format(angle),     (xtext,ytext+7*htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("s  = {}").format(scale),     (xtext,ytext+8*htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv.putText(imgDemo,("dif.= {}").format(deltaC),   (xtext,ytext+9*htext), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0))


        cv.ellipse(imgDemo, (center, box, angle), SFpaletteOCV.DarkBrown, 1, cv.LINE_AA)

        xa1 = int(np.round(-box[0]/2*math.cos(angle*np.pi/180)+center[0]))
        ya1 = int(np.round(-box[0]/2*math.sin(angle*np.pi/180)+center[1]))
        xa2 = int(np.round( box[0]/2*math.cos(angle*np.pi/180)+center[0]))
        ya2 = int(np.round( box[0]/2*math.sin(angle*np.pi/180)+center[1]))
        
        xb1 = int(np.round(-box[1]/2*math.cos((90+angle)*np.pi/180)+center[0]))
        yb1 = int(np.round(-box[1]/2*math.sin((90+angle)*np.pi/180)+center[1]))
        xb2 = int(np.round( box[1]/2*math.cos((90+angle)*np.pi/180)+center[0]))
        yb2 = int(np.round( box[1]/2*math.sin((90+angle)*np.pi/180)+center[1]))

        if box[0]>box[1]:
            cv.line(imgDemo,(xa1,ya1),(xa2,ya2),SFpaletteOCV.Brown,2, cv.LINE_AA)
            cv.line(imgDemo,(xb1,yb1),(xb2,yb2),SFpaletteOCV.Brown,1, cv.LINE_AA)
        elif box[0]<box[1]:
            cv.line(imgDemo,(xa1,ya1),(xa2,ya2),SFpaletteOCV.Brown,1, cv.LINE_AA)
            cv.line(imgDemo,(xb1,yb1),(xb2,yb2),SFpaletteOCV.Brown,2, cv.LINE_AA)
        else:
            cv.line(imgDemo,(xa1,ya1),(xa2,ya2),SFpaletteOCV.Brown,2, cv.LINE_AA)
            cv.line(imgDemo,(xb1,yb1),(xb2,yb2),SFpaletteOCV.Brown,2, cv.LINE_AA)

        w, q = zip(*sorted(zip(weights, points)))
        palette = Viewer.CreatePalette()
        N = len(q)
        for n in range(N):
            p = (round(q[n][0]), round(q[n][1]))
            if w[n]<0:
                color = SFpaletteOCV.LightBlue 
            else:
                palId = math.floor(w[n]*255)
                color = palette[palId]
            cv.circle(imgDemo, p, 1, color, -1)

        barw = 30
        xpal = barCorner[0]-barw*2
        ypal = ytext #barCorner[1]
        yofs = 4
        cv.putText(imgDemo,("w"), (xpal+round(barw/3),ypal-2*yofs), cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.putText(imgDemo,("1.0"), (xpal-barw,ypal+yofs), cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.putText(imgDemo,("0.5"), (xpal-barw,ypal+127+yofs), cv.FONT_HERSHEY_PLAIN, 1, 0)
        for col in reversed(palette):
            cv.line(imgDemo, (xpal,ypal), (xpal+barw,ypal), col, 1, cv.LINE_AA)
            ypal += 1
        cv.putText(imgDemo,("0.0"), (xpal-barw,ypal+yofs), cv.FONT_HERSHEY_PLAIN, 1, 0)

        #cv.imwrite('imgDemo.png', imgDemo)

        #cv.namedWindow(title, cv.WINDOW_NORMAL)
        cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
        cv.imshow(title, imgDemo)
        cv.waitKey(40)
        #cv.destroyWindow(title)

        

    @staticmethod
    def CreatePalette() -> list:
        #colormap={cmap}{color={FewDarkPurple} color={FewLightOrange} color=(FewDarkBlue)},
        palette = []
        col0 = SFpaletteOCV.DarkRed
        col1 = SFpaletteOCV.LightYellow
        col2 = SFpaletteOCV.DarkGreen

        for n in range(256):
            
            if n<128:
                v = n/128
                col = (col0[0]*(1-v) + v*col1[0], col0[1]*(1-v) + v*col1[1], col0[2]*(1-v) + v*col1[2])
            else:
                v = (n-128)/128
                col = (col1[0]*(1-v) + v*col2[0], col1[1]*(1-v) + v*col2[1], col1[2]*(1-v) + v*col2[2])
            palette.append(col)
        return palette