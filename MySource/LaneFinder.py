'''
Created on Sep 11, 2017

@author: andre
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob, os



'''
This class is used for the purpose to identify yellow and white structures on an image 
and perform the following transformations
RGB -> GRAY
RGB -> HSV
HSV -> GRAY 
'''
class ColorTransformations(object):
    def ConvertBGRImageToGrayColorSpace(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def ConvertImageToHSVSpace(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def ConvertHSVImageToGrayColorSpace(self, image):
        image1 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return self.ConvertBGRImageToGrayColorSpace(image1)

    def YellowHSVMask(self,image):
        yellowHSVLow  = np.array([ 0, 80, 200])
        yellowHSVHigh = np.array([ 40, 255, 255])
        return cv2.inRange(image,yellowHSVLow, yellowHSVHigh)
        
    def WhiteHSVMask(self,image):
        whiteHSVLow  = np.array([  20,   0,   200])
        whiteHSVHigh = np.array([ 255,  80, 255])
        return cv2.inRange(image,whiteHSVLow, whiteHSVHigh)
    
    def ApplyWhiteAnYellowColorMasks(self, image):
        yellowMask = self.YellowHSVMask(image)
        whiteMask = self.WhiteHSVMask(image)
        mask = cv2.bitwise_or(yellowMask, whiteMask)
        filtered = cv2.bitwise_and(image, image, mask = mask)
        return filtered




class LaneRecognition(object):
    '''
     Constructor 
     '''
    def __init__(self):
        self.Color = ColorTransformations

    def ApplyCannyEdgeDetection(self, image, lowerThreshold=50, upperThreshold=150):
        return cv2.Canny(image, lowerThreshold, upperThreshold)
        
    def ApplyGaussianSmoothing(self, image, kernelSize =5):
        return cv2.GaussianBlur(image,(kernelSize, kernelSize),0)
    
    
    def YTetragonBorder(self, yImageSize, yRegionRatio):
        return yImageSize - (int (float (yImageSize) * yRegionRatio))
    
    '''
    Generates a tetragon-mask based on the image size and the following two inputs 
    yRegionRatio: Gives the ratio of the considered image in y direction, which will be used to build the tetragon  
    xQuadTopWidth: Gives the full width of the tetragon at the top. BottomWidth is full x-range of image  
    The mask is applied bitwise to the input-image and the result is returned
    '''
    def DefineTetragonROIAndApplyToImage(self, image, yRegionRatio, xTetragonTopWidth = 80):
        yImage = image.shape[0]
        xImage = image.shape[1]
        mask = np.zeros_like(image)
        xTopLeft = xImage/2 - xTetragonTopWidth/2
        xTopRight = xImage/2 + xTetragonTopWidth/2
        xBottomLeft = 0
        xBottomRight = xImage
        yBottom = yImage
        yTetragonBorder = self.YTetragonBorder(yImage, yRegionRatio)
        tetragonEdges = np.array([[(xBottomLeft, yBottom), (xTopLeft, yTetragonBorder), (xTopRight, yTetragonBorder), (xBottomRight, yBottom)]], dtype= np.int32)
        cv2.fillPoly(mask, tetragonEdges, 255)
        result = cv2.bitwise_and(image,mask)
        return result
    
    
    '''
     Angular resolution converted in radians, input in degrees
    '''
    def ProbalisticHoughTransform(self, image, distanceResolution =2, angularResolution=1, numberOfPointsDefiningALine = 15, minimalLineLength = 20, maximalLineGap=20):
        angularResolution = angularResolution*np.pi/180.0
        identifiedLines = cv2.HoughLinesP(image, distanceResolution, angularResolution, numberOfPointsDefiningALine, np.array([]), minimalLineLength, maximalLineGap)
        return identifiedLines
    
    '''
    '''
    def CreateLineImage(self, lines, emptyImage):
        for line in lines:
            for x1,y1,x2,y2 in line:
                '''
                Check if length of lines is a good discriminator 
                Answer: No
                x = x1 - x2
                y = y1 - y2
                res = (x**2 + y**2)**.5
                if(res > 120):
                    cv2.line(emptyImage,(x1,y1),(x2,y2),(0,0,255),10)
                elif(res > 80):
                    cv2.line(emptyImage,(x1,y1),(x2,y2),(0,255,0),10)
                else:
                    cv2.line(emptyImage,(x1,y1),(x2,y2),(0,0,255),10)
                '''
                cv2.line(emptyImage,(x1,y1),(x2,y2),(0,0,255),10)
        return emptyImage
                
    
    '''
    Reduce the number of lines which are not in the direction of the lanes.
    Central angle is choosen to be 45 degree. Default gap is 15 degree 
    '''
    def ApplyAngleFiltering(self, lines, centralAngle = 45, angleGap=15):
        lowerAngle = math.radians(centralAngle-angleGap)
        upperAngle = math.radians(centralAngle+angleGap) 
        filteredLines = []
        if lines is None:
            return filteredLines
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2-y1)/(x2-x1)
                angle = abs(math.atan(slope))
                if(angle > lowerAngle and angle < upperAngle):
                    filteredLines.append( [[x1, y1, x2, y2]] )
        return filteredLines

    
    '''
    We have two lanes and consequently two categories for all lines: 
    a) Lines with negative slope represent the left lane (coordinate system of image)
    b) Lines with positive slope represent the right lane
    Does a linear assumption using the mean values of left/right lines work?
    Answer: NO, not good enough. Resulting lanes are flickering/unstable and detection in video 'challenge.mp4' is bad
    '''
    def IdentifyLanes(self, lines, yImageLowerBorder, xImageRightBorder, yRegionRatio):
        leftLane = [[0,0,0,0]]
        rightLane = [[0,0,0,0]]
        leftLaneSlope = 0
        rightLaneSlope = 0
        
        leftLaneLineSlopes = [] 
        rightLaneLineSlopes = []
        rightLaneLineIntercepts = []
        leftLaneLineIntercepts = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if(abs(x2-x1) < 1.e-3):
                    continue
                slope = float(y2-y1)/float(x2-x1)
                xCenter = (int )(xImageRightBorder/2)
                if( slope < 0):
                    leftLaneLineSlopes.append(slope)
                    leftLaneLineIntercepts.append(y1-slope*x1)
                    
                elif(slope >0):
                    rightLaneLineSlopes.append(slope)
                    rightLaneLineIntercepts.append(y1-slope*x1)
                    
        # Calculate the mean values of slope and intercept for both lanes  
        if(len(leftLaneLineSlopes) > 0):
            leftLaneSlope = np.mean(leftLaneLineSlopes)
            leftLaneIntercept = np.mean(leftLaneLineIntercepts)
        if(len(rightLaneLineSlopes) > 0):
            rightLaneSlope = np.mean(rightLaneLineSlopes)
            rightLaneIntercept = np.mean(rightLaneLineIntercepts)
         
        yTetragonBorder = self.YTetragonBorder(yImageLowerBorder, yRegionRatio)
        
        y1Left = yImageLowerBorder
        y1Right = y1Left
        y2Left = (int) (yTetragonBorder)
        y2Right = y2Left
        
        if(abs(leftLaneSlope) > 1.e-8 ):
            x1Left = (int) ((yImageLowerBorder-leftLaneIntercept)/leftLaneSlope)
            x2Left = (int) ((yTetragonBorder-leftLaneIntercept)/leftLaneSlope)
            leftLane = [[x1Left, y1Left, x2Left, y2Left]]
            
        if(abs(rightLaneSlope) > 1.e-8 ):
            x1Right = (int) ((yImageLowerBorder-rightLaneIntercept)/rightLaneSlope)
            x2Right = (int) ((yTetragonBorder-rightLaneIntercept)/rightLaneSlope)
            rightLane = [[x1Right, y1Right, x2Right, y2Right]]
            
        
        return [leftLane, rightLane]
    
    
    def CalculateLanePoints(self, x,y,degree):
        try:
            laneFit = np.polyfit(x,y,degree)
            laneFunction = np.poly1d(laneFit)
            return laneFunction(x).astype(int)
        except TypeError:
            return np.ndarray([0])
    
    '''
    Using polynomial fit instead of simple linear fit
    - Use points of all lines and assign them according to the x-Value to left and right lane
    - Make fit and define function
    Result is not yet satisfying, since the right lane segments are not connected.
    '''
    def IdentifyLinesWithPolynom(self, lines, xImageRightBorder, polynomDegree =2):
        leftLaneLineX = []
        leftLaneLineY = []
        rightLaneLineX = []
        rightLaneLineY = []
        xCenter = (int )(xImageRightBorder/2)
        leftLanePoints = np.column_stack(( [], [] )).reshape(-1,1,2)
        rightLanePoints = np.column_stack(( [], [] )).reshape(-1,1,2)
        if(lines is None):
            return leftLanePoints, rightLanePoints

        for line in lines:
            for x1, y1, x2, y2 in line:
                if( x1 < xCenter and x2 < xCenter):
                    leftLaneLineX.append(x1)
                    leftLaneLineY.append(y1)
                    leftLaneLineX.append(x2)
                    leftLaneLineY.append(y2)

                elif( x1 > xCenter and x2 > xCenter):
                    rightLaneLineX.append(x1)
                    rightLaneLineY.append(y1)
                    rightLaneLineX.append(x2)
                    rightLaneLineY.append(y2)

        
        newLeftLaneY = self.CalculateLanePoints(leftLaneLineX, leftLaneLineY, polynomDegree)
        newRightLaneY= self.CalculateLanePoints(rightLaneLineX, rightLaneLineY, polynomDegree)
        if not newLeftLaneY.any():
            leftLaneLineX = []
        if not newRightLaneY.any():
            rightLaneLineX = []
        leftLanePoints = np.column_stack(( np.asarray(leftLaneLineX), newLeftLaneY )).reshape(-1,1,2)
        rightLanePoints = np.column_stack((np.asarray(rightLaneLineX), newRightLaneY)).reshape(-1,1,2)
        
        return leftLanePoints, rightLanePoints 
        
    '''
    Create not connected Polygon with cv2 
    '''
    def CreatePolygonLines(self, image, leftPoints, rightPoints):
        image1 = cv2.polylines(image,[leftPoints],False,(0,0,255),10)
        return cv2.polylines(image1,[rightPoints],False,(0,0,255),10)
        
    



    def ProcessImage(self, image, linear):
        yRegionRatio = 0.4
        '''
        ColorTransformations to identify white and yellow objects
        '''
        color = self.Color
        hsv = color().ConvertImageToHSVSpace(image)
        whiteAndYellow = color().ApplyWhiteAnYellowColorMasks(hsv)
        gray = color().ConvertHSVImageToGrayColorSpace(whiteAndYellow)
        #gray = color().ConvertBGRImageToGrayColorSpace(image)
        
        '''
        Smoothing, CannyEdge, ROI
        '''  
        smoothie = self.ApplyGaussianSmoothing(gray)
        edges = self.ApplyCannyEdgeDetection(smoothie)
        roi = self.DefineTetragonROIAndApplyToImage(edges, yRegionRatio = yRegionRatio)
        
        '''
        Line identification via HoughTransform and extrapolation.
        Two methods are possible (depending on the video): 
        - Simple linear fit with angle filtering
        - Polynomial fit for challenge-video
        Goal must be to use one global function or a decision maker to decide when to use which function
        '''        
        identifiedLines = self.ProbalisticHoughTransform(roi)
        emptyImage = np.copy(image)*0 # Empty frame for lines
        #lineImage = self.CreateLineImage(identifiedLines, emptyImage)
        
        if(linear):
            identifiedLines = self.ApplyAngleFiltering(identifiedLines)
            twoLanes = self.IdentifyLanes(identifiedLines, emptyImage.shape[0], emptyImage.shape[1], yRegionRatio = yRegionRatio)
            lineImage = self.CreateLineImage(twoLanes, emptyImage)
        else:
            leftPoints, rightPoints = self.IdentifyLinesWithPolynom(identifiedLines, emptyImage.shape[1], polynomDegree=2)
            lineImage = self.CreatePolygonLines(emptyImage, leftPoints, rightPoints)
        
        result = cv2.addWeighted(image, 0.8, lineImage, 1, 0)
        
        return result
    
    

    
    
    
class ImageLaneFinder(object):
    def __init__(self, path):
        self.Processor = LaneRecognition()
        self.Path = path
    
    def LaneImages(self):
        os.chdir(self.Path)
        self.ImagePaths = [file for file in glob.glob("*.jpg")]
        for path in self.ImagePaths:
            image = cv2.imread(self.Path+path)
            result = self.Processor.ProcessImage(image, True)
            cv2.imshow('frame', result)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break

            
            
        
        
class VideoLaneFinder(object):
    def __init__(self, path):
        self.Path = path
        self.Processor = LaneRecognition()

    def LaneVideo(self):
        os.chdir(self.Path)
        self.VideoPaths = [file for file in glob.glob("*.mp4")]

        for path in self.VideoPaths:        
            cap = cv2.VideoCapture(self.Path+path)
            linear = False
            count =0
            if 'solidYellowLeft.mp4' in path or 'solidWhiteRight.mp4' in path:
                linear = True
                continue
            while(cap.isOpened()):
                ret, frame = cap.read()
                count +=1
                if(ret):
                    result = self.Processor.ProcessImage(frame, linear)
                    cv2.imshow('frame', result)
                    
                    #cv2.imwrite(ExportPath+'challengeMine'+str(count)+'.jpg',result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:   
                    break
            cap.release()
            cv2.destroyAllWindows()
    


DefaultPath = './../test_videos/'
DefaultImagePath = './../test_images/'


'''
This shows all images
'''
#AllImages = ImageLaneFinder(DefaultImagePath)
#AllImages.LaneImages()


'''
This shows all videos
'''
AllVideos = VideoLaneFinder(DefaultPath)
AllVideos.LaneVideo()

