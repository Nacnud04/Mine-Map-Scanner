from pytesseract import *
import argparse
import cv2
import sys
import numpy as np
from math import *
from datetime import datetime
from os.path import exists
import time

st = time.time()


# generate arguments to take via command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
        required=True,
        help="path to input image to be OCR'd")
ap.add_argument("-d", "--display",
        required=False,
        help="image to display output upon")
ap.add_argument("-dg", "--divide-grid",
        required=False,
        help="grid to divide image based on. is a tuple. ex: (rr, cc)")
ap.add_argument("-de", "--divide-exact",
        required=False, type=list, default = [],
        help="divide image based on a pixel perfect grid")
ap.add_argument("-sa", "--start-angle",
        required=False, type = int, default = -90,
        help="the angle at which the image will start at")
ap.add_argument("-fa", "--final-angle",
        required = False, type = int, default = 90,
        help = "the angle which the image will end at")
ap.add_argument("-af", "--angle-fidelity",
        required = False, type = float, default=5,
        help = "the incriment at which the angle of the image will increase")
ap.add_argument("-lr", "--line-removal",
        required = False, type = bool, default=True,
        help="enable/disable line detection & removal")
ap.add_argument("-c", "--min-conf",
        type=int, default=0,
        help="minimum confidence value to filter weak text detection")
ap.add_argument("-r", "--min-ratio",
        type=float, default = 1,
        help="minimum ratio allowed; calculated by width/height")
ap.add_argument("-mw", "--min-width",
        type=int, default = 0,
        help="minimum width allowed")
ap.add_argument("-mh", "--min-height",
        type=int, default = 0,
        help="minimum height allowed")
ap.add_argument("-v", "--view-type",
        type=str, default='',
        help="type of input image. n = normal, i = dark")
ap.add_argument("-oh", "--output-height",
        type=int, default=720,
        help="output height of the final image, maintains aspect ratio")
ap.add_argument("-f", "--apply-filters",
        type=str, default = 'n',
        help="applies certain filters onto the image")
ap.add_argument("-oi", "--override-invert",
        type=bool, default = False,
        help="override automatic invert detection")
ap.add_argument("-ir", "--intersection-requirement",
        type=float, default = 0.05,
        help="minimum percentage intersection for bounding boxes to merge")
ap.add_argument("-rc", "--result-consolidation",
        type=bool, default = True,
        help="enable/disable result consolidation")
args = vars(ap.parse_args())

# interpret arguments given
imagepath, displaypath = args["image"], args["display"]
gridstring, pixeldivisions = args["divide_grid"], args["divide_exact"]
startangle, finalangle, anglefidelity = args["start_angle"], args["final_angle"], args["angle_fidelity"]
minconf = args["min_conf"]
minratio, minwidth, minheight = args["min_ratio"], args["min_width"], args["min_height"]
viewtype, outputheight = args["view_type"], args["output_height"]
appliedfilters = args["apply_filters"]

# check inputs

# cant have a % intersect above 1 or less than 0
if args["intersection_requirement"] <= 1 and args["intersection_requirement"] >= 0:
    pass
else:
    raise Exception("Percentage of intersection cannot be greater than 1, or less than 0")
# cannot have a window height of 0
if outputheight <= 0:
    raise Exception("cannot have a window height less than or equal to 0")
# cannot have a confidence above 100
if minconf > 100:
    raise Exception("cannot have confidence above 100%")
# angle fidelity of 0 would cause the program to infinitely run
if anglefidelity == 0:
    raise Exception("angle fidelity cannot be 0")
# check to see if file exists
if exists(imagepath) == False:
    raise Exception(f"image {imagepath} not found")

# split image path into folder and filename
components = imagepath.split('/')
imagename = components[-1]
components.pop(-1)
folder = ""
for ele in components:
    folder += f'{ele}/'
print(f'Image Name: {imagename} \nFolder Path: {folder}')

# interpret grid divisions
if gridstring:
    grid = gridstring.split(",")
    grid = [int(x) for x in grid]


### BEGIN PRE-PROCESSING ###
from PreProcessing import *
from PostProcessing import *

initialfilters = []

# apply black and white filter to speed up processes
initialfilters.append('gray')

# check to see if invert required
inversion = detectInversion(folder, imagename)
if inversion == True and args['override_invert'] == False:
    initialfilters.append('invert')

# enable high pass filter with contrast
initialfilters.append("highpass")
initialfilters.append("contrast")

# default to removing lines
if args['line_removal'] == True:
    initialfilters.append('nolines')

# add image filters.
# parse given filters.
def parseFilters(appliedfilters):
    filterlist = []
    filterlist[:0] = appliedfilters
    return filterlist
filterlist = parseFilters(appliedfilters)

## SET GRID SEGMENTATION OR CONTOUR SEGMENTATION
segmenttype = "grid"

# Initalize image, and cut into segments
image = Image(imagepath, initialfilters)
if segmenttype == "grid":
    ppdh, ppdw, maxi, maxj = image.genSegments(grid)
    # calculate the number of images which are being processed
    h, w = grid[0], grid[1]
    try :
        numimages = int((((finalangle - startangle) / anglefidelity) + 1) * (h*2-1)*(w*2-1) * len(filterlist))
    except ZeroDivisionError:
        raise Exception("angle fidelity is 0. the program must be stopped or it will run continously.")
elif segmenttype == "contour":
    bordersize = 0.25
    segmentnum, origins, offsets = image.genCropContours(bordersize)
    # calculate number of images being processed
    try:
        numimages = int((((finalangle - startangle) / anglefidelity) + 1) * (segmentnum) * len(filterlist))
    except ZeroDivisionError:
        raise Exception("angle fidelity is 0. the program must be stopped or it will run continously.")

### MAIN PROCESS ###
import OCR as ocr

import threading
import sys

print(f'\nNumber of images being processed: {numimages}')

def grabSegmentGrid(folder, segment):
    fullname = f'{folder}{segment[0]}-{segment[1]}.jpg'
    image = cv2.imread(fullname)
    return image

def grabSegmentContour(folder, num):
    fullname = f'{folder}{num}.jpg'
    image = cv2.imread(fullname)
    return image

# this function rotates the input image about it's center
def rotateImage(img, theta):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    #output = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    output = rotated
    return output

if segmenttype == "grid":
    # cycle through segments
    cycles = 1
    data = []
    for i in range(grid[0] * 2 - 1):
        for j in range(grid[1] * 2 - 1):
            img =  grabSegmentGrid(folder, (i, j))

            # cycle through angles
            theta = startangle
            while theta <= finalangle:

                sys.stdout.write(f'\rScanned: {cycles}/{numimages}  |  Name: {folder}{i}-{j}.jpg    ')
                sys.stdout.flush()
                rotated = rotateImage(img, theta)
                newdata = ocr.OCR(rotated, theta, minheight, minwidth, minconf, minratio, (i, j))
                if newdata != []:
                    data.extend(newdata)

                theta += anglefidelity
                cycles += 1
elif segmenttype == "contour":
    data = []
    cycles = 0
    for i in range(segmentnum):
        img = grabSegmentContour(folder, i)
        xorigin, yorigin = origins[i]
        xoffset, yoffset = offsets[i]
        theta = startangle
        while theta <= finalangle:
            # apply OCR here
            sys.stdout.write(f'\rScanned: {cycles}/{numimages}  |  Name: {folder}{i}.jpg    ')
            sys.stdout.flush()
            rotated = rotateImage(img, theta)
            newdata = ocr.OCR(rotated, theta, minheight, minwidth, minconf, 
                              minratio, (i), xorigin=xorigin, yorigin=yorigin,
                              xoffset=xoffset, yoffset=yoffset)
            if newdata != []:
                data.extend(newdata)
            theta += anglefidelity
            cycles += 1
        

### FINAL PROCESS ###

img = cv2.imread(imagepath)

# finalize data
if segmenttype == "grid":
    transformed = transformDataGrid(data, folder, ppdh, ppdw)
elif segmenttype == "contour":
    transformed = transformDataContoured(data, folder, bordersize)

# merge data
transformed = mergeIntersecting(transformed, args["intersection_requirement"])

# export data
exportData(folder, 'out.csv', transformed)

# cleanup data
if segmenttype == "grid":
    cleanupGrid(folder, maxi, maxj)
elif segmenttype == "contour":
    cleanupContour(folder, segmentnum)

# calculate total runtime

def secondsConversion(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 6

    return "%d:%02d:%02d" % (hour, minutes, seconds)

et = time.time()
elapsed = et - st
print(secondsConversion(elapsed))
print(f'\nDetected {len(transformed)} words.')

# show data
showData(transformed, img, outputheight, 6)
