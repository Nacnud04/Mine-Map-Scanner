'''
This programs function is as follows:
Segment the image according to a given grid
'''

from inspect import Attribute
from os import remove
import cv2
import numpy as np
import sys
import math
import time
import matplotlib.pyplot as plt
import scipy
from pylsd import lsd
from PostProcessing import *

# this function defines where the image is to be segmented based on a grid
def defSegmentsGrid(img, divisions):
    # size of array being equal to divisions[0] * 2 + 1 allows for overlapping crops.
    array = np.zeros((divisions[0] * 2 + 1, divisions[1] * 2 + 1, 2), dtype=int)
    try :
        (height, width) = img.shape[:2]
    except AttributeError:
        raise Exception("File not found")
    # ppd - pixels per division
    ppdh, ppdw = int(height / (divisions[0] * 2)), int(width / (divisions[1] * 2))
    for i in range(divisions[0] * 2 + 1):
        for j in range(divisions[1] * 2 + 1):
            array[i, j, 0] = int(i * ppdh)
            array[i, j, 1] = int(j * ppdw)
    return array, ppdh, ppdw

# this function defines where the image is to be segmented based on specific coordinates
def defSegmentsExact(img, vertdivisions, hordivisions):
    array = np.zeros((len(vertdivisions), len(hordivisions), 2), dtype=int)
    for i in range(len(vertdivisions)):
        for j in range(len(hordivisions)):
            array[i, j, 0] = vertdivisions[i]
            array[i, j, 1] = hordivisions[j]
    return array

# detect if needs inversion
def detectInversion(folder, filename):
    fullname = f'{folder}/{filename}'
    img = cv2.imread(fullname)
    invert = False
    average = np.average(img)
    if average <= 1:
        if average <= 0.5:
            invert = True
    elif average >= 1:
        if average <= 128:
            invert = True
    return invert

# never ever use this, will literially take hours on a large image
def contrastFunction(alpha):
    constant = math.log(256)/128
    if alpha >= 129:
        alpha = 255
    else:
        alpha = math.e**(alpha * constant) - 1
    return alpha

# always use this is extremely fast and awesome in every way.
def contrastFunctionFast(array):
    constant = math.log(256)/128
    array = math.e**(array*constant)-1
    return array

# sends the dark peak to 0 and the lightpeak to 255 while following an exponential curve
def contrastPeaks(array, darkpeak, valley, lightpeak):
    # first blur the image to remove fibers
    #array = cv2.GaussianBlur(array, (3, 3), 3)

    # offsets
    darkoffset = 50
    lightoffset = -50

    darkpeak = darkpeak + darkoffset
    lightpeak = lightpeak + lightoffset

    # apply contrast function
    const = math.log(256)/(lightpeak - darkpeak)
    A = math.e**(-1*darkpeak*(const))
    array = A*math.e**(array*const)-1
    array = np.clip(array, 0, 255)
    return array

# IMAGE CLASS (10/20/22)
class Image:

    def __init__(self, filepath, initialfilters):

        self.raw = cv2.imread(filepath)
        self.data = self.raw

        # split image path into folder and filename
        components = filepath.split('/')
        imagename = components[-1]
        components.pop(-1)
        folder = ""
        for ele in components:
            folder += f'{ele}/'

        self.path = folder
        self.name = imagename

        self.applyInitialFilters(initialfilters)


    # convert to gray
    def convertToGray(self):
        if type(self.data) != np.ndarray:
            raise Exception(f'Input image data in function convertToGray was not of type <class \'np.ndarray\'> and instead was {type(self.data)}.')
        rgb2k = np.array([0.114, 0.587, 0.299])
        matrix_int = np.round(np.sum(self.data * rgb2k, axis=-1)).astype('uint8')
        # matrix_float = np.sum(imagedata * rgb2k, axis=-1) / 255
        self.data = matrix_int

    # do a bw inversion
    def invert(self):
        maxvalue = np.full(self.data.shape, 256)
        self.data = np.subtract(maxvalue, self.data)
        print(f'BW color inversion required')

    def guessBitDepth(self):
        maximum = np.max(self.raw)
        minimum = np.min(self.raw)
        print(f'maxiumum bit value: {maximum}, minimum bit val: {minimum}')
        print(f'array size: {np.shape(self.data)}')

    def highPassGaussian(self):
        # constant blur width:21, height:21, 
        # only works with uint8, apparently all other files entering except the one are uint32????
        # maybe inverting the data converts it to uint32??
        print(f'Applying high pass filter')
        if self.data.dtype != np.uint8:
            self.data = self.data.astype(np.uint8)
        highpass = self.data - cv2.GaussianBlur(self.data, (21, 21), 3) + 127
        return highpass

    def increaseContrast(self, type):
        print(f'Applying contrast filter...')
        if type == "stdfast":
            self.data = contrastFunctionFast(self.data)
        elif type == "slow":
            contrast_v = np.vectorize(contrastFunction)
            self.data = contrast_v(self.data)
        elif type == "peaks":
            self.data = contrastPeaks(self.data, self.darkpeak,
                                     self.valley, self.lightpeak)
            self.data = self.data.astype(np.uint8)

    # line removal
    def removeLines(self):
        imagedata = self.data
        segments = lsd(imagedata, scale=0.5, ang_th=20)
        characteristics = np.zeros(shape=(segments.shape[0], 5))

        # interpet + fuse segments
        angles = []
        for i in range(segments.shape[0]):
            pt1 = (int(segments[i, 0]), int(segments[i, 1]))
            pt2 = (int(segments[i, 2]), int(segments[i, 3]))

            # calculate length
            length = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

            # calculate angle of segment
            try:
                angle = math.atan((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))
            except ZeroDivisionError:
                angle = math.pi / 2
                continue

            angles.append(angle)
            characteristics[i] = [length, angle, 0, 0, 0]

        segments = np.append(segments, characteristics, axis = 1)

        ### DEFINE MINIMUM LENGTH FOR A LINE TO BE KEPT
        minlength = 55

        badsegments = []

        # remove short segments
        for i in range(segments.shape[0]):
            pt1 = (int(segments[i, 0]), int(segments[i, 1]))
            pt2 = (int(segments[i, 2]), int(segments[i, 3]))
            width, length = segments[i, 4], segments[i, 5]
        
            if length <= minlength:
                badsegments.append(i)

        lines = Lines(segments, badsegments)
        mergesegments = lines.mergeLines()


        self.data = lines.hideLines(imagedata, 1.25)

    def applyInitialFilters(self, initialfilters):
        for filtertype in initialfilters:
            if filtertype == 'invert':
                self.invert()
            elif filtertype == 'gray':
                self.convertToGray()
            elif filtertype == 'nolines':
                self.removeLines()
            elif filtertype == "highpass":
                self.highPassGaussian()
            elif filtertype == "contrast":
                self.captureHistogram(12, False)
                self.increaseContrast("peaks")

    def captureHistogram(self, window, show):
        hist, bins = np.histogram(self.data.ravel(), bins=256)
        factor = np.ones(window)/window # create an array of 1/value
        hist = np.convolve(hist, factor, mode="same") # capture rolling average
        peak_indices = scipy.signal.find_peaks_cwt(hist, 24)
        betweenmin = np.min(np.array(hist[peak_indices[0] : peak_indices[-1]])) # capture minimum value in between peaks
        if show == True:
            plt.plot(range(len(hist)), hist, label = "Histogram")
            plt.scatter(peak_indices, np.zeros(len(peak_indices)), label = "Peaks")
            plt.scatter(np.where(np.array(hist[peak_indices[0] : peak_indices[-1]]) == betweenmin)+peak_indices[0], [0], color='red', label = 'Valley')
            plt.legend()
            plt.show()

        self.darkpeak = peak_indices[0]
        self.lightpeak = peak_indices[-1]
        self.valley = betweenmin


    def genCropContours(self, bordersize):
        print(f'Generating contours')
        contour = Contours(self.data)
        contour.reformatContours()
        cropbounds = contour.cropLocations()
        num, origins, offsets = self.cropByContours(cropbounds, bordersize)
        #contour.showContours(self.raw)
        return num, origins, offsets

    def genSegments(self, divisions):
        maxi, maxj = 0, 0
        array, ppdh, ppdw = defSegmentsGrid(self.data, divisions)
        for i in range(np.shape(array)[0] - 1):
            for j in range(np.shape(array)[1] - 1):
                y0, x0 = array[i, j]
                skip = False
                try :
                    y1, x1 = array[i + 2, j + 2]
                except IndexError:
                    if divisions[0] % 2 == 0:
                        skip = True
                    y1, x1 = array[i + 1, j + 1]
                if i == 0:
                    try :
                        y1, x1 = array[i + 2, j + 2]
                    except IndexError:
                        if divisions[0] % 2 == 0:
                            skip = True
                        y1, x1 = array[i + 1, j + 1]
                elif j == 0:
                    try:
                        y1, x1 = array[i + 2, j + 2]
                    except IndexError:
                        if divisions[0] % 2 == 0:
                            skip = True
                        y1, x1 = array[i + 1, j + 1]
                if skip == False:
                    crop = self.data[y0:y1, x0:x1]
                    sys.stdout.flush()
                    sys.stdout.write(f'\rSegmenting Image... |i:{i}|j:{j}|y0:{y0}|y1:{y1}|x0:{x0}|x1:{x1}     ')
                    cv2.imwrite(f'{self.path}/{i}-{j}.jpg', crop)
                    if i >= maxi:
                        maxi = i
                    if j >= maxj:
                        maxj = j
        return ppdh, ppdw, maxi, maxj

    def cropByContours(self, cropbounds, bordersize = 0.25):
        i = 0
        origins = []
        offsets = []
        for cropbound in cropbounds:
            xmin, ymin, xmax, ymax = cropbound
            origins.append((xmin, ymin))
            crop = self.data[ymin:ymax, xmin:xmax]
            # add white borders to allow not cropping text during a rotation
            crop = cv2.copyMakeBorder(
                        crop,
                        top=int(bordersize * abs(ymin-ymax)),
                        bottom=int(bordersize * abs(ymin-ymax)),
                        left=int(bordersize * abs(xmin-xmax)),
                        right=int(bordersize * abs(xmin-xmax)),
                        borderType=cv2.BORDER_CONSTANT,
                        value=255
                    )
            offsets.append((int(0.25*abs(xmin-xmax)), int(0.25*abs(ymin-ymax))))
            cv2.imwrite(f'{self.path}/{i}.jpg', crop)
            sys.stdout.flush()
            sys.stdout.write(f'\rSegmenting Image... |i:{i}|y0:{ymin}|y1:{ymax}|x0:{xmin}|x1:{xmax}     ')
            i += 1
        return i, origins, offsets

class Contours:

    def __init__(self, image):
        
        # generate contours in init
        ret, th1 = cv2.threshold(image, 28, 255, cv2.THRESH_BINARY_INV)
        bw = th1.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) # reduces size of detected rectangles
        dilation = cv2.dilate(bw, kernel, iterations = 14) # joins contours
        self.cnts, self.heirarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # finds contours

    def reformatContours(self):
        listcontours = []
        for contour in self.cnts:
            contourpoints = np.array([[0, 0]])
            for point in contour:
                contourpoints = np.vstack([contourpoints, point[0]])
            contourpoints = np.delete(contourpoints, 0, axis = 0)
            listcontours.append(contourpoints)
        listcontours = np.array(listcontours)
        self.reformated = listcontours

    def cropLocations(self):
        cropboundslist = []
        for contour in self.reformated:
            cropbounds = np.array(self.bbox2D(contour))
            if self.removeSmallSquares(cropbounds) != None and self.removeSmallSides(cropbounds) != None:
                cropboundslist.append(cropbounds)
        return np.array(cropboundslist)

    def bbox2D(self, array):
        xs = array[:, 0]
        ys = array[:, 1]
        minx, maxx = np.min(xs), np.max(xs)
        miny, maxy = np.min(ys), np.max(ys)
        return minx, miny, maxx, maxy

    def cleanByRatio(self, bounds):
        if abs(bounds[2] - bounds[3]) / abs(bounds[0] - bounds[1]) < 0.08 or abs(bounds[0] - bounds[1]) / abs(bounds[2] - bounds[3]) > 20:
            return None
        else:
            return True

    def removeSmallSquares(self, bounds):
        dx = abs(bounds[0] - bounds[2])
        dy = abs(bounds[1] - bounds[3])
        if dy/dx >= 0.95 and dy/dx <= 1.05:
            if dy <= 31 or dx <= 31:
                return None
        return True

    def removeSmallSides(self, bounds):
        dx = abs(bounds[0] - bounds[2])
        dy = abs(bounds[1] - bounds[3])
        if dy <= 40 or dx <= 40:
            return None
        else:
            return True

    def showContours(self, img, outputheight=800):
        contoured = cv2.drawContours(img, self.cnts, -1, (0, 255, 0), 3)
        cv2.imshow("Contours", ResizeWithAspectRatio(contoured, height=outputheight))
        cv2.waitKey(0)
        
    def getContours(self):
        return self.cnts

    def getHeirarchy(self):
        return self.heirarchy


class Lines:

    def __init__(self, segments, badsegments):
        self.segmentsraw = segments
        self.badsegments = badsegments
        self.segments = []

        for i in range(segments.shape[0]):
            if i not in badsegments:
                self.segments.append(segments[i])
        self.segments = np.array(self.segments)

        # one segment follows this order:
        # x1, y1, x2, y2, width, length, angle, 0, 0, 0

        self.sortConstants(math.pi/90, 1)

    def getSegments(self):
        return self.segments

    def sortConstants(self, tautheta, epsilons):
        self.tautheta = tautheta
        self.epsilons = epsilons

    def taus(self, segment):
        taus = self.epsilons * segment[5]
        return taus

    # sorts lines by decreasing length
    def sortLength(self):
        sortorder = np.argsort(self.segments[:, 5])
        sortorder = np.flip(sortorder)
        self.segments = self.segments[sortorder]
        return self.segments

    def similarAngle(self, segment):
        segmentindex = np.where(self.segments==segment) #[0] because there should only be one index
        possible = []
        for i in range(len(self.segments)):
            if np.any(segmentindex == i) == False:
                if abs(segment[6]-self.segments[i][6]) <= self.tautheta:
                    possible.append(i)
        return possible

    def similarX(self, segment, spatialprox):
        segmentindex = np.where(self.segments==segment) #[0] because there should only be one index
        possible = []
        for i in range(len(self.segments)):
            if np.any(segmentindex == i) == False:
                append = False
                if abs(segment[0]-self.segments[i][0]) <= spatialprox:
                    append = True
                elif abs(segment[0]-self.segments[i][2]) <= spatialprox:
                    append = True
                elif abs(segment[2]-self.segments[i][0]) <= spatialprox:
                    append = True
                elif abs(segment[2]-self.segments[i][2]) <= spatialprox:
                    append = True
                if append == True:
                    possible.append(i)
        return possible

    def similarY(self, segment, spatialprox):
        segmentindex = np.where(self.segments==segment) #[0] because there should only be one index
        possible = []
        for i in range(len(self.segments)):
            if np.any(segmentindex == i) == False:
                append = False
                if abs(segment[1]-self.segments[i][1]) <= spatialprox:
                    append = True
                elif abs(segment[1]-self.segments[i][3]) <= spatialprox:
                    append = True
                elif abs(segment[3]-self.segments[i][1]) <= spatialprox:
                    append = True
                elif abs(segment[3]-self.segments[i][3]) <= spatialprox:
                    append = True
                if append == True:
                    possible.append(i)
        return possible

    def mergeTwoLines(self, segment1, segment2):

        # find the longer segment
        longer = 0
        if segment1[5] > segment2[5]:
            longer = 1
        else:
            longer = 2

        # find d (shortest distance between two end points)
        d = 10000
        for i in [0, 2]:
            for j in [0, 2]:
                newd = math.sqrt((segment1[i] - segment2[j])**2+(segment1[i+1] + segment2[j+1])**2)
                if newd < d:
                    d = newd

        # compute and compare spatial threshold
        # also compute normalized lengths
        if longer == 1:
            spatialthreshold = self.taus(segment1)
            normalizedlength = segment2[5] / segment1[5]
        else:
            spatialthreshold = self.taus(segment2)
            normalizedlength = segment1[5] / segment2[5]

        # if distance is too large stop merging
        if d > spatialthreshold:
            return None

        # compute normalized d
        normalizedd = d / spatialthreshold

        # compute penalty
        penalty = normalizedlength + normalizedd

        # compute angle threshold
        angthresh = (1 - (1/(1+math.e**(-2*(penalty-1.5)))))*self.tautheta

        # find angle difference
        # angles are in radians
        deltatheta = abs(segment1[6] - segment2[6])
        
        if deltatheta < angthresh or deltatheta > (math.pi - angthresh):

            # capture merged line endpoints
            # find the greatest distance between endpoints
            bigdistance = 0
            points = [0, 0, 0, 0]
            for i in [0, 2]:
                for j in [0, 2]:
                    newd = math.sqrt((segment1[i] - segment2[j])**2+(segment1[i+1] - segment2[j+1])**2)
                    if newd > bigdistance:
                        bigdistance = newd
                        points = [segment1[i], segment1[i + 1], segment2[j], segment2[j+1]]
            # if big line is longer than dist between points, then the endpoints are the endpoints of the original line
            if longer == 1:
                if segment1[5] > bigdistance:
                    points = [segment1[0], segment1[1], segment1[2], segment1[3]]
                    bigdistance = segment1[5]
            else:
                if segment2[5] > bigdistance:
                    points = [segment2[0], segment2[1], segment2[2], segment2[3]]
                    bigdistance = segment2[5]
            # calculate angle of new line
            try:
                angle = math.atan((points[3] - points[1]) / (points[2] - points[0]))
            except ZeroDivisionError:
                angle = math.pi / 2
            # if angle difference is greater than half threshold angle dont merge
            if abs(segment1[6] - angle) > 0.5*angthresh:
                return None
            else:
                points.extend([segment1[4], bigdistance, angle, 0, 0, 0])
                return points

    def mergeLines(self):
        n = len(self.segments) # repeat the function until every single segment has run through (hence why segment list length is needed)
        self.segments = self.sortLength() # sort the list of segments by length as the larger segments have a greater accuracy
        initiallength = len(self.segments)
        
        # remove duplicate lines
        #self.segments = np.unique(self.segments, axis = 0)
        i = 0
        for segment in self.segments:
            spatialthreshold = self.taus(segment)
            anglepossible = self.similarAngle(segment) # create list of possible segments within a certain angle.
            xpossible = self.similarX(segment, spatialthreshold)
            ypossible = self.similarY(segment, spatialthreshold)
            #sys.stdout.flush()
            indexes = []
            for possible in anglepossible: # only keep the segments which are possible in all categories.
                if possible in xpossible:
                    if possible in ypossible:
                        indexes.append(int(possible))
            #sys.stdout.write(f'\rMerging detected lines... Ang#:{len(anglepossible)}|X#{len(xpossible)}|Y#{len(ypossible)}|TOT#{len(indexes)}        ')
            toremove = []
            for index in indexes:
                merged = self.mergeTwoLines(segment, self.segments[index])
                if merged != None:
                    sys.stdout.flush()
                    sys.stdout.write(f'\rMerging detected lines...')
                    segment = merged
                    toremove.append(int(index))
            iterations = 0
            for j in toremove:
                self.segments = np.delete(self.segments, j-iterations, axis = 0)
                iterations += 1 # this works since the smallest indexes are subtracted first
            i += 1
        print(f'\n# of line segments reduced from {initiallength} to {len(self.segments)}')
        return self.segments

    def showAngles(self, segments, centerangles, angles):
        x, y = [], []
        for i in range(segments.shape[0]):
            degangle = int(math.degrees(segments[i, 6]))
            if degangle not in x:
                x.append(degangle)
                y.append(0)
        for i in range(max(x)-min(x)):
            if i not in x:
                x.append(i)
                y.append(0)
        # order data
        x.sort()
        print(f'Angles Min: {min(angles)}  |  Angles Max: {max(angles)}')
        for i in range(segments.shape[0]):
            degangle = int(math.degrees(segments[i, 6]))
            index = x.index(degangle)
            y[index] += 1

        for angle in centerangles:
            xangle = [angle, angle]
            yangle = [0, max(y)]
            plt.plot(xangle, yangle, linestyle='dashdot', color='lightgray', linewidth='1')
        plt.plot(x, y)
        plt.xlabel("Angle")
        plt.ylabel("Count")
        plt.title("Angle frequency")
        plt.show()

    def hideLines(self, img, widthmodifier):
        try:
            imgr = img[:, :, :1]
            imgg = img[:, :, 1:2]
            imgb = img[:, :, 2:]
            red, green, blue = np.median(imgr), np.median(imgg), np.median(imgb)
            for i in range(self.segments.shape[0]):
                pt1 = (int(self.segments[i, 0]), int(self.segments[i, 1]))
                pt2 = (int(self.segments[i, 2]), int(self.segments[i, 3]))
                width, length, angle = self.segments[i, 4], self.segments[i, 5], self.segments[i, 6]
                width = width * widthmodifier
                if length >= 80:
                    # prevents lines in text from being removed.
                    cv2.line(img, pt1, pt2, (red, green, blue), int(np.ceil(width)))
        except IndexError:
            # assume array is 2D (ie, grayscale, non rgb)
            median = np.median(img)
            for i in range(self.segments.shape[0]):
                pt1 = (int(self.segments[i, 0]), int(self.segments[i, 1]))
                pt2 = (int(self.segments[i, 2]), int(self.segments[i, 3]))
                width, length, angle = self.segments[i, 4], self.segments[i, 5], self.segments[i, 6]
                width = width * widthmodifier
                if length >= 80:
                    # prevents lines in text from being removed.
                    cv2.line(img, pt1, pt2, median, int(np.ceil(width)))
                    #cv2.line(img, pt1, pt2, 128, int(np.ceil(width)))
        return img
